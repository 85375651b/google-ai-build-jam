import test from 'node:test';
import assert from 'node:assert/strict';
import { JSDOM } from 'jsdom';
import { build } from 'esbuild';
import { createRequire } from 'node:module';
import { mkdtemp, writeFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { pathToFileURL, fileURLToPath } from 'node:url';

const require = createRequire(import.meta.url);
const output = await build({ entryPoints: [fileURLToPath(new URL('../../src/App.jsx', import.meta.url))],
  bundle: true, platform: 'node', format: 'esm', write: false,
  plugins: [{ name: 'shared-react', setup(builder) { builder.onResolve({ filter: /^react$/ }, () => ({ path: require.resolve('react'), external: true })); } }],
});
const directory = await mkdtemp(join(tmpdir(), 'worldbuilding-ui-'));
const moduleFile = join(directory, 'App.mjs');
await writeFile(moduleFile, output.outputFiles[0].text);

test('actual React UI changes Stop to post-processing, then Send before delayed EOF', async () => {
  const dom = new JSDOM('<div id="root"></div>', { url: 'http://localhost/' });
  const previous = {};
  for (const key of ['window', 'document', 'navigator', 'localStorage', 'HTMLElement', 'IS_REACT_ACT_ENVIRONMENT', 'fetch']) previous[key] = Object.getOwnPropertyDescriptor(globalThis, key);
  for (const key of ['window', 'document', 'navigator', 'localStorage', 'HTMLElement']) Object.defineProperty(globalThis, key, { value: dom.window[key], configurable: true });
  globalThis.IS_REACT_ACT_ENVIRONMENT = true;
  dom.window.HTMLElement.prototype.scrollIntoView = () => {};
  localStorage.setItem('wb_onboarding_dismissed', 'true');
  const React = await import('react');
  const { createRoot } = await import('react-dom/client');
  const { default: App } = await import(pathToFileURL(moduleFile));
  let streamController, requestId;
  let storedMessages = [];
  const encoder = new TextEncoder();
  globalThis.fetch = async (url, options = {}) => {
    if (url.endsWith('/stream')) {
      requestId = JSON.parse(options.body).request_id;
      return new Response(new ReadableStream({ start(c) { streamController = c; } }));
    }
    if (url === '/api/system/status') return Response.json({ local_model: 'gpt-oss:20b', local_model_status: { ready: true }, reference_documents: [] });
    if (url === '/api/worlds') return Response.json([{ id: 'world-1', name: 'Test world' }]);
    if (url === '/api/worlds/world-1') return Response.json({ id: 'world-1', name: 'Test world', expertise_profiles: [] });
    if (url.endsWith('/messages')) return Response.json(storedMessages);
    if (url.endsWith('/conversations')) return Response.json([{ id: 'conversation-1', title: 'Main' }]);
    if (url.endsWith('/consistency')) return Response.json({ relationships: [], contradictions: [], needs_review: [] });
    return Response.json([]);
  };
  const root = createRoot(document.getElementById('root'));
  async function settle(predicate) {
    for (let n = 0; n < 30; n++) {
      await React.act(async () => { await new Promise((resolve) => setTimeout(resolve, 5)); });
      if (predicate()) return;
    }
    assert.fail('UI did not reach expected state');
  }
  const buttons = () => [...document.querySelectorAll('button')].map((button) => button.textContent.trim());
  function send(type, data = {}) {
    if (type === 'complete') storedMessages = [{ id: data.user_message_id, role: 'USER', content: 'Planet X is called Pallene.' }, data.assistant_message];
    streamController.enqueue(encoder.encode(`data: ${JSON.stringify({ type, request_id: requestId, conversation_id: 'conversation-1', world_id: 'world-1', ...data })}\n\n`));
  }
  try {
    await React.act(async () => root.render(React.createElement(App)));
    await settle(() => document.querySelector('textarea.chat-textarea'));
    const textarea = document.querySelector('textarea.chat-textarea');
    await React.act(async () => {
      Object.getOwnPropertyDescriptor(dom.window.HTMLTextAreaElement.prototype, 'value').set.call(textarea, 'Planet X is called Pallene.');
      textarea.dispatchEvent(new dom.window.Event('input', { bubbles: true }));
    });
    await React.act(async () => document.querySelector('form.chat-form').dispatchEvent(new dom.window.Event('submit', { bubbles: true, cancelable: true })));
    await settle(() => buttons().includes('Stop'));
    await React.act(async () => { send('generation_started'); send('delta', { text: 'Recorded Pallene.' }); send('answer_completed', { text: 'Recorded Pallene.' }); send('postprocessing_started', { message: 'Updating World Bible…' }); });
    assert.ok(!buttons().includes('Stop'));
    assert.ok(buttons().includes('Updating World Bible…'));
    assert.ok(buttons().includes('Cancel update'));
    await React.act(async () => send('complete', {
      answer_saved: true, user_message_id: 'user-1',
      assistant_message: { id: 'assistant-1', role: 'ASSISTANT', content: 'Recorded Pallene.', metadata: {
        canon_result: { status: 'committed', operations: [
          { state: 'committed', entity_ref: 'Planet X', property_key: 'canonical_name', value: 'Pallene' },
          { state: 'proposed', operation: { entity_ref: 'Planet Y', value: 'Neso' } },
        ] },
      } },
    }));
    await settle(() => buttons().includes('Send'));
    assert.ok(!buttons().includes('Stop'));
    assert.ok(document.body.textContent.includes('Recorded Pallene.'));
    assert.ok(document.body.textContent.includes('World Bible: Saved'));
    assert.ok(document.body.textContent.includes('Planet Y → Neso — Awaiting approval'));
    // The controlled stream deliberately never calls close().
  } finally {
    await React.act(async () => root.unmount());
    dom.window.close();
    for (const [key, descriptor] of Object.entries(previous)) {
      if (descriptor) Object.defineProperty(globalThis, key, descriptor);
      else delete globalThis[key];
    }
    await rm(directory, { recursive: true, force: true });
  }
});
