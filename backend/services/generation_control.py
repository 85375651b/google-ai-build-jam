"""Per-request cooperative cancellation and an async-I/O bridge for existing services."""
import asyncio
import concurrent.futures
from contextvars import ContextVar
import threading

current_generation = ContextVar('current_generation', default=None)


class GenerationCancelled(BaseException):
    """Must bypass legacy Exception-based model fallbacks."""


class GenerationControl:
    def __init__(self, request_id, conversation_id, world_id):
        self.request_id = request_id
        self.conversation_id = conversation_id
        self.world_id = world_id
        self.loop = asyncio.get_running_loop()
        self.events = asyncio.Queue(maxsize=256)
        self.done = asyncio.Event()
        self.lock = threading.RLock()
        self.cancelled = threading.Event()
        self.futures = set()
        self.io_tasks = set()
        self.reason = 'Cancelled by the author.'
        self.error_code = None
        self.state = 'accepted'
        self.terminal = None
        self.answer = None
        self.task = None

    def check(self):
        if self.cancelled.is_set():
            raise GenerationCancelled()

    def cancel(self, reason='Cancelled by the author.', error_code=None):
        with self.lock:
            if self.terminal is not None:
                return False
            if not self.cancelled.is_set():
                self.reason = reason
                self.error_code = error_code
            self.cancelled.set()
            self.state = 'cancelling'
            for future in tuple(self.futures):
                future.cancel()
            return True

    def wait_for(self, coroutine):
        """The worker waits; the main event loop owns and cancels real network I/O."""
        with self.lock:
            if self.cancelled.is_set():
                coroutine.close()
                raise GenerationCancelled()
            future = concurrent.futures.Future()

            def schedule():
                if future.cancelled():
                    coroutine.close()
                    return
                entered = False

                async def owned():
                    nonlocal entered
                    entered = True
                    return await coroutine

                task = self.loop.create_task(owned())
                self.io_tasks.add(task)

                def completed(task):
                    self.io_tasks.discard(task)
                    if not entered:
                        coroutine.close()
                    with self.lock:
                        if future.done():
                            # Retrieve exceptions even if the worker was already cancelled.
                            if not task.cancelled():
                                task.exception()
                            return
                        if task.cancelled():
                            future.cancel()
                        elif task.exception() is not None:
                            future.set_exception(task.exception())
                        else:
                            future.set_result(task.result())

                task.add_done_callback(completed)
                future.add_done_callback(lambda value: self.loop.call_soon_threadsafe(task.cancel) if value.cancelled() else None)

            self.loop.call_soon_threadsafe(schedule)
            self.futures.add(future)
        try:
            return future.result()
        except concurrent.futures.CancelledError:
            raise GenerationCancelled() from None
        finally:
            with self.lock:
                self.futures.discard(future)

    def emit(self, event_type, **data):
        self.check()
        event = dict(data, type=event_type, request_id=self.request_id,
                     conversation_id=self.conversation_id, world_id=self.world_id)
        self.wait_for(self.events.put(event))

    def finish(self, event):
        # Called only after the terminal journal transaction commits.
        with self.lock:
            self.terminal = event
            self.state = event['type']

    async def drain_io(self):
        await asyncio.sleep(0)
        pending = list(self.io_tasks)
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
