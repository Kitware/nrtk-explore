import multiprocessing
import asyncio
import signal
import threading
import logging
import queue
import uuid
from .predictor import Predictor


def _child_worker(request_queue, result_queue, model_name, force_cpu):
    signal.signal(signal.SIGINT, signal.SIG_IGN)  # Ignore Ctrl+C in child
    logger = logging.getLogger(__name__)
    predictor = Predictor(model_name=model_name, force_cpu=force_cpu)

    while True:
        try:
            msg = request_queue.get()
        except (EOFError, KeyboardInterrupt):
            logger.debug("Worker: Exiting on interrupt or queue EOF.")
            break
        if msg is None:  # Exit signal
            logger.debug("Worker: Received EXIT command. Shutting down.")
            break

        command = msg["command"]
        req_id = msg["req_id"]
        payload = msg.get("payload", {})
        logger.debug(f"Worker: Received {command} with ID {req_id}")

        if command == "SET_MODEL":
            try:
                predictor = Predictor(
                    model_name=payload["model_name"],
                    force_cpu=payload["force_cpu"],
                )
                result_queue.put((req_id, {"status": "OK"}))
            except Exception as e:
                logger.exception("Failed to set model.")
                result_queue.put((req_id, {"status": "ERROR", "message": str(e)}))
        elif command == "INFER":
            try:
                predictions = predictor.eval(payload["images"])
                result_queue.put((req_id, {"status": "OK", "result": predictions}))
            except Exception as e:
                logger.exception("Inference failed.")
                result_queue.put((req_id, {"status": "ERROR", "message": str(e)}))
        elif command == "RESET":
            try:
                predictor.reset()
                result_queue.put((req_id, {"status": "OK"}))
            except Exception as e:
                logger.exception("Reset failed.")
                result_queue.put((req_id, {"status": "ERROR", "message": str(e)}))

    logger.debug("Worker: shutting down.")


class MultiprocessPredictor:
    def __init__(self, model_name="facebook/detr-resnet-50", force_cpu=False):
        self._lock = threading.Lock()
        self.model_name = model_name
        self.force_cpu = force_cpu
        self._proc = None
        self._request_queue = None
        self._result_queue = None
        self._start_process()

        def handle_shutdown(signum, frame):
            self.shutdown()

        signal.signal(signal.SIGINT, handle_shutdown)

    def _start_process(self):
        with self._lock:
            if self._proc is not None and self._proc.is_alive():
                self.shutdown()
            multiprocessing.set_start_method("spawn", force=True)
            self._request_queue = multiprocessing.Queue()
            self._result_queue = multiprocessing.Queue()
            self._proc = multiprocessing.Process(
                target=_child_worker,
                args=(
                    self._request_queue,
                    self._result_queue,
                    self.model_name,
                    self.force_cpu,
                ),
                daemon=True,
            )
            self._proc.start()

    def set_model(self, model_name, force_cpu=False):
        with self._lock:
            self.model_name = model_name
            self.force_cpu = force_cpu
            req_id = str(uuid.uuid4())
            self._request_queue.put(
                {
                    "command": "SET_MODEL",
                    "req_id": req_id,
                    "payload": {
                        "model_name": self.model_name,
                        "force_cpu": self.force_cpu,
                    },
                }
            )
            return self._wait_for_response(req_id)

    async def infer(self, images):
        if not images:
            return {}
        with self._lock:
            req_id = str(uuid.uuid4())
            new_req = {"command": "INFER", "req_id": req_id, "payload": {"images": images}}
            self._request_queue.put(new_req)

        resp = await self._wait_for_response_async(req_id)
        return resp.get("result")

    async def _wait_for_response_async(self, req_id):
        return await asyncio.get_event_loop().run_in_executor(None, self._get_response, req_id, 40)

    def _wait_for_response(self, req_id):
        return self._get_response(req_id, 40)

    def _get_response(self, req_id, timeout=40):
        while True:
            try:
                r_id, data = self._result_queue.get(timeout=timeout)
            except queue.Empty:
                raise TimeoutError("No response from worker.")
            if r_id == req_id:
                return data

    def reset(self):
        with self._lock:
            req_id = str(uuid.uuid4())
            self._request_queue.put({"command": "RESET", "req_id": req_id})
            return self._wait_for_response(req_id)

    def shutdown(self):
        with self._lock:
            try:
                self._request_queue.put(None)
            except Exception:
                logging.warning("Could not send exit message to worker.")
            if self._proc:
                self._proc.join()
