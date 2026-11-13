import test from 'node:test';
import assert from 'node:assert/strict';
import { GenerationClient, isPostprocessingStage } from '../../src/generationClient.js';

const encoder = new TextEncoder();
const id = 'request-1';
const scope = { conversationId: 'conversation-1', worldId: 'world-1', content: 'A message', requestId: id };
const event = (type, extra = {}, requestId = id) => ({ type, request_id: requestId, conversation_id: scope.conversationId, world_id: scope.worldId, ...extra });
const complete = (requestId = id) => event('complete', { answer_saved: true, user_message_id: 'user-1', assistant_message: { id: 'answer-1', content: 'A complete answer.' } }, requestId);
const frame = (value) => `data: ${JSON.stringify(value)}\n\n`;
const tick = () => new Promise((resolve) => setTimeout(resolve, 0));
function harness() {
  let controller;
  const chunks = new ReadableStream({ start(value) { controller = value; } });
  const events = [], states = [], failures = [], calls = [];
  const client = new GenerationClient({
    fetchImpl: async (url, options) => {
      calls.push([url, options]);
      if (url.endsWith('/cancel')) return Response.json({ terminal: event('cancelled', { message: 'Stopped.' }) });
      return new Response(chunks);
    },
    onEvent: (value) => events.push(value), onState: (value) => states.push(value),
    onFailure: (value) => failures.push(value), newId: () => id,
  });
  return { client, events, states, failures, calls, controller: () => controller };
}

test('complete is terminal even if the connection never reaches EOF', async () => {
  const h = harness();
  const running = h.client.start(scope);
  h.controller().enqueue(encoder.encode(frame(event('delta', { text: 'A complete answer.' })) + frame(complete())));
  assert.equal(await Promise.race([running, new Promise((_, reject) => setTimeout(() => reject(Error('waited for EOF')), 200))]), true);
  assert.equal(h.client.active, null);
  assert.equal(h.states.at(-1), 'complete');
  assert.equal(h.events.at(-1).type, 'complete');
});

test('visible completion enters saving/post-processing rather than idle', async () => {
  const h = harness(); const running = h.client.start(scope);
  h.controller().enqueue(encoder.encode(frame(event('generation_started')) + frame(event('answer_completed', { text: 'Done.' }))));
  await tick();
  assert.equal(h.client.active.stage, 'answer_completed');
  assert.equal(isPostprocessingStage(h.client.active.stage), true);
  h.controller().enqueue(encoder.encode(frame(event('postprocessing_started', { message: 'Updating World Bible…' }))));
  await tick(); assert.equal(h.client.active.stage, 'postprocessing');
  h.controller().enqueue(encoder.encode(frame(complete()))); await running;
  assert.equal(h.client.active, null);
});

for (const separator of ['\n\n', '\r\n\r\n', '\r\r']) {
  test(`handles split UTF-8 and SSE frames (${JSON.stringify(separator)})`, async () => {
    const h = harness(); const running = h.client.start(scope);
    const raw = encoder.encode(frame(event('delta', { text: '世界' })).replaceAll('\n\n', separator) + frame(complete()).replaceAll('\n\n', separator));
    for (const byte of raw) h.controller().enqueue(Uint8Array.of(byte));
    await running;
    assert.equal(h.events[0].text, '世界'); assert.equal(h.states.at(-1), 'complete');
  });
}

test('processes a final complete JSON event buffered at EOF without blank separator', async () => {
  const h = harness(); const running = h.client.start(scope);
  h.controller().enqueue(encoder.encode(frame(complete()).trimEnd())); h.controller().close();
  await running; assert.equal(h.states.at(-1), 'complete');
});

for (const body of ['', frame(event('delta', { text: 'partial' })), 'data: {broken', 'data: null\n\n', frame(event('complete'))]) {
  test(`rejects malformed or incomplete streams: ${body.slice(0, 35)}`, async () => {
    const h = harness(); const running = h.client.start(scope);
    if (body) h.controller().enqueue(encoder.encode(body)); h.controller().close();
    await running; assert.equal(h.states.at(-1), 'error'); assert.equal(h.failures.length, 1); assert.equal(h.client.active, null);
  });
}

test('backend error is terminal immediately, without EOF', async () => {
  const h = harness(); const running = h.client.start(scope);
  h.controller().enqueue(encoder.encode(frame(event('error', { message: 'Persistence failed.' }))));
  await running; assert.equal(h.states.at(-1), 'error'); assert.equal(h.events.at(-1).message, 'Persistence failed.');
});

test('interrupted connection is a failure and requests backend cancellation', async () => {
  const h = harness(); const running = h.client.start(scope);
  h.controller().error(Error('Connection lost')); await running; await tick();
  assert.equal(h.failures.length, 1); assert.ok(h.calls.some(([url]) => url.endsWith('/cancel')));
});

test('Stop immediately after submission is idempotent and targets its request ID', async () => {
  const h = harness(); const running = h.client.start(scope);
  await Promise.all([h.client.cancel(), h.client.cancel()]);
  // A real fetch rejects its reader on abort; release this controlled reader as well.
  h.controller().close(); await running;
  assert.equal(h.calls.filter(([url]) => url.endsWith('/cancel')).length, 1);
  assert.ok(h.calls.some(([url]) => url.includes(`/requests/${id}/cancel`)));
  assert.equal(h.states.at(-1), 'cancelled');
});

for (const destination of [{ conversationId: 'conversation-2', worldId: 'world-1' }, { conversationId: 'conversation-2', worldId: 'world-2' }]) {
  test(`scope switch and old cleanup cannot change a newer request: ${JSON.stringify(destination)}`, async () => {
    const h = harness();
    const old = h.client.start(scope); h.client.detach();
    const pending = [];
    h.client.fetchImpl = async (url) => url.endsWith('/cancel')
      ? Response.json({ terminal: event('cancelled', {}, 'new-id') })
      : new Response(new ReadableStream({ start(c) { pending.push(c); } }));
    const newer = h.client.start({ ...scope, ...destination, requestId: 'new-id' });
    h.controller().close(); await old;
    assert.equal(h.client.active.id, 'new-id');
    assert.equal(h.client.active.conversationId, destination.conversationId);
    pending[0].enqueue(encoder.encode(JSON.stringify({}) /* force a nonterminal EOF */)); pending[0].close(); await newer;
    assert.ok(!h.events.some((value) => value.request_id === id));
  });
}

test('duplicate submissions do not issue a second POST', async () => {
  const h = harness(); const running = h.client.start(scope);
  assert.equal(await h.client.start(scope), false);
  assert.equal(h.calls.length, 1);
  h.controller().enqueue(encoder.encode(frame(complete()))); await running;
});

test('idle watchdog clears the owner and cancels stalled work', async () => {
  const h = harness(); h.client.idleMs = 15;
  const running = h.client.start(scope); await new Promise((resolve) => setTimeout(resolve, 30));
  assert.equal(h.client.active, null); assert.equal(h.states.at(-1), 'error');
  assert.ok(h.calls.some(([url]) => url.endsWith('/cancel')));
  h.controller().close(); await running;
});

test('heartbeats keep a slow but observable operation alive', async () => {
  const h = harness(); h.client.idleMs = 30; const running = h.client.start(scope);
  for (let n = 0; n < 5; n++) { await new Promise((r) => setTimeout(r, 10)); h.controller().enqueue(encoder.encode(frame(event('heartbeat')))); }
  assert.ok(h.client.active);
  h.controller().enqueue(encoder.encode(frame(complete()))); await running;
  assert.equal(h.failures.length, 0);
});

test('an event with a different owner is rejected', async () => {
  const h = harness(); const running = h.client.start(scope);
  h.controller().enqueue(encoder.encode(frame(complete('other-id')))); h.controller().close(); await running;
  assert.equal(h.events.length, 0); assert.equal(h.failures.length, 1);
});

test('an explicit HTTP rejection does not cancel another existing request', async () => {
  const calls = [];
  const client = new GenerationClient({ fetchImpl: async (url) => {
    calls.push(url); return Response.json({ detail: { message: 'Already active.' } }, { status: 409 });
  } });
  await client.start(scope);
  assert.equal(calls.length, 1);
  assert.equal(client.active, null);
});

test('queued generation progress cannot replace the cancelling state', async () => {
  const h = harness(); let resolveCancellation;
  h.client.fetchImpl = async (url) => url.endsWith('/cancel') ? await new Promise((r) => { resolveCancellation = r; }) : new Response(new ReadableStream({ start(c) { h.stream = c; } }));
  const running = h.client.start(scope); await tick();
  const stopping = h.client.cancel();
  h.stream.enqueue(encoder.encode(frame(event('generation_started')))); await tick();
  assert.equal(h.client.active.stage, 'cancelling');
  resolveCancellation(Response.json({ terminal: event('cancelled', { message: 'Stopped.' }) }));
  await stopping; h.stream.close(); await running;
  assert.equal(h.states.at(-1), 'cancelled');
});

test('a terminal event stops parsing without waiting for or processing trailing bytes', async () => {
  const h = harness(); const running = h.client.start(scope);
  h.controller().enqueue(encoder.encode(frame(complete()) + 'data: {bad trailing data}\n\n'));
  assert.equal(await running, true);
  assert.equal(h.states.at(-1), 'complete');
  assert.equal(h.failures.length, 0);
});
