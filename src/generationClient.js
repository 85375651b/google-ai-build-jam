// One owner per request; transport parsing and lifecycle are testable without React.
export const TERMINAL_EVENTS = new Set(['complete', 'cancelled', 'error']);
const EVENT_TYPES = new Set(['request_started', 'progress', 'generation_started', 'turn_meta', 'delta',
  'answer_completed', 'saving_started', 'response_saved', 'postprocessing_started', 'heartbeat', ...TERMINAL_EVENTS]);
export const isBusyStage = (stage) => !['idle', 'complete', 'cancelled', 'error'].includes(stage);
export const isPostprocessingStage = (stage) => ['answer_completed', 'saving', 'postprocessing', 'committing'].includes(stage);

export function createSSEParser(onEvent) {
  const decoder = new TextDecoder('utf-8', { fatal: true });
  let buffer = '';
  let data = [];
  let stopped = false;
  function dispatch() {
    if (stopped || !data.length) return;
    const raw = data.join('\n');
    data = [];
    let event;
    try { event = JSON.parse(raw); } catch { throw new Error('The server sent malformed stream data.'); }
    if (!event || typeof event !== 'object' || !EVENT_TYPES.has(event.type)) {
      throw new Error('The server sent an invalid lifecycle event.');
    }
    if (onEvent(event) === false) stopped = true;
  }
  function line(value) {
    if (value === '') dispatch();
    else if (value.startsWith('data:')) data.push(value.slice(5).replace(/^ /, ''));
    // SSE comments and event/id/retry fields do not contain application payloads.
  }
  let previousCR = false;
  function feed(text) {
    for (const char of text) {
      if (stopped) return;
      if (char === "\n" && previousCR) { previousCR = false; continue; }
      previousCR = char === "\r";
      if (char === "\r" || char === "\n") { line(buffer); buffer = ""; }
      else buffer += char;
    }
  }
  return {
    push(chunk) { feed(decoder.decode(chunk, { stream: true })); },
    finish() {
      feed(decoder.decode());
      if (buffer) { line(buffer); buffer = ""; }
      dispatch(); // A final valid event need not have a trailing blank line.
    },
  };
}

export class GenerationClient {
  constructor({ fetchImpl = (...args) => fetch(...args), onEvent = () => {}, onState = () => {},
    onFailure = () => {}, idleMs = 45000, cancelMs = 12000, newId = () => crypto.randomUUID() } = {}) {
    Object.assign(this, { fetchImpl, onEvent, onState, onFailure, idleMs, cancelMs, newId });
    this.active = null;
  }

  owns(owner) { return this.active === owner && !owner.finished; }

  _state(owner, stage, message = '') {
    if (!this.owns(owner) || (owner.stage === 'cancelling' && stage !== 'cancelling')) return;
    owner.stage = stage;
    this.onState(stage, message, owner);
  }

  _watch(owner) {
    clearTimeout(owner.watchdog);
    owner.watchdog = setTimeout(() => {
      if (!this.owns(owner)) return;
      const error = new Error('The connection stopped reporting progress. Cancellation was requested; reload the conversation before retrying.');
      owner.failure = error;
      void this._cancelBackend(owner).catch(() => {});
      owner.controller.abort();
      this._fail(owner, error);
    }, owner.idleMs);
  }

  _finish(owner, event) {
    if (!this.owns(owner)) return;
    owner.finished = true;
    clearTimeout(owner.watchdog);
    this.active = null; // Release synchronously, before waiting for any network cleanup.
    this.onEvent(event, owner);
    if (!this.active) this.onState(event.type, event.message || '', owner);
  }

  _fail(owner, error) {
    if (!this.owns(owner)) return;
    owner.finished = true;
    clearTimeout(owner.watchdog);
    this.active = null;
    this.onFailure(error, owner);
    if (!this.active) this.onState('error', error.message, owner);
  }

  _event(owner, event) {
    if (!this.owns(owner)) return;
    if (event.request_id !== owner.id || event.conversation_id !== owner.conversationId || event.world_id !== owner.worldId) {
      throw new Error('Received an event for a different request or conversation.');
    }
    if (event.type === 'complete' && (!event.answer_saved || !event.user_message_id ||
      !event.assistant_message?.id || typeof event.assistant_message.content !== 'string')) {
      throw new Error('The server sent an incomplete completion event.');
    }
    if (event.type === 'delta' && typeof event.text !== 'string') throw new Error('The server sent an invalid text chunk.');
    if (event.type === 'request_started' && Number.isFinite(event.client_idle_seconds)) {
      owner.idleMs = Math.max(15000, event.client_idle_seconds * 1000);
    }
    this._watch(owner);
    if (TERMINAL_EVENTS.has(event.type)) {
      this._finish(owner, event);
      return;
    }
    const stages = {
      request_started: 'accepted', generation_started: 'generating', answer_completed: 'answer_completed',
      saving_started: 'saving', response_saved: 'saving', postprocessing_started: 'postprocessing',
    };
    if (event.type === 'progress') this._state(owner, event.stage || owner.stage, event.message);
    else if (stages[event.type]) this._state(owner, stages[event.type], event.message);
    this.onEvent(event, owner);
  }

  async start({ conversationId, worldId, content, requestId, tempId }) {
    if (this.active) return false; // Synchronous ownership closes the React double-submit race.
    const owner = { id: requestId || this.newId(), conversationId, worldId, content, tempId,
      controller: new AbortController(), finished: false, idleMs: this.idleMs, stage: 'accepted' };
    this.active = owner;
    this._state(owner, 'accepted', 'Request accepted…');
    this._watch(owner);
    let reader;
    try {
      const response = await this.fetchImpl(`/api/conversations/${conversationId}/stream`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ content, request_id: owner.id }), signal: owner.controller.signal,
      });
      if (!response.ok) {
        owner.rejected = true;
        const error = await response.json().catch(() => ({}));
        throw new Error(error?.detail?.message || `Request failed (HTTP ${response.status}).`);
      }
      if (!response.body) throw new Error('The server did not open a response stream.');
      reader = response.body.getReader();
      const parser = createSSEParser((event) => {
        this._event(owner, event);
        return this.owns(owner);
      });
      while (this.owns(owner)) {
        const { value, done } = await reader.read();
        if (!this.owns(owner)) break;
        if (done) {
          parser.finish();
          if (this.owns(owner)) throw new Error('The response stream ended without a completion or cancellation event. Reload the conversation before retrying.');
          break;
        }
        parser.push(value);
      }
      return true;
    } catch (error) {
      if (this.owns(owner)) {
        if (!owner.rejected) void this._cancelBackend(owner).catch(() => {});
        this._fail(owner, owner.failure || error);
      }
      return false;
    } finally {
      // Do not wait for EOF or a slow cancel() promise after a terminal event.
      owner.controller.abort();
      if (reader) {
        void Promise.resolve(reader.cancel()).catch(() => {});
        reader.releaseLock();
      }
      clearTimeout(owner.watchdog);
      // An old owner's cleanup must not clear a newer active request.
    }
  }

  async _cancelBackend(owner) {
    if (owner.cancelPromise) return owner.cancelPromise;
    const timeout = new AbortController();
    const timer = setTimeout(() => timeout.abort(), this.cancelMs);
    owner.cancelPromise = this.fetchImpl(`/api/conversations/${owner.conversationId}/requests/${owner.id}/cancel`, {
      method: 'POST', signal: timeout.signal, keepalive: true,
    }).then(async (response) => {
      if (!response.ok) throw new Error('Unable to confirm cancellation. Reload the conversation before retrying.');
      return response.json();
    }).finally(() => clearTimeout(timer));
    return owner.cancelPromise;
  }

  async cancel() {
    const owner = this.active;
    if (!owner) return;
    this._state(owner, 'cancelling', 'Cancelling the request…');
    try {
      const result = await this._cancelBackend(owner);
      if (this.owns(owner) && result.terminal) {
        this._event(owner, result.terminal);
        owner.controller.abort();
      }
    } catch (error) {
      owner.controller.abort();
      this._fail(owner, error);
    }
  }

  detach() {
    const owner = this.active;
    if (!owner) return;
    owner.finished = true;
    clearTimeout(owner.watchdog);
    this.active = null;
    void this._cancelBackend(owner).catch(() => {});
    owner.controller.abort();
  }
}
