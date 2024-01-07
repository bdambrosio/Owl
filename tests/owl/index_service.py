from fastapi import FastAPI, BackgroundTasks
from starlette.concurrency import run_in_threadpool
from starlette.config import Config
from starlette.datastructures import CommaSeparatedStrings
import asyncio
import traceback

from typing import Any
import subprocess
import json

app = FastAPI()
job_id=0
import asyncio
import json

class PersistentQueue(asyncio.Queue):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.persistence_file = "indexing_service_queue.json"
        self._items_list = []  # Mirror list for serialization

    async def put(self, item):
        await super().put(item)
        self._items_list.append(item)  # Keep the list in sync

    async def get(self):
        item = await super().get()
        self._items_list.remove(item)  # Keep the list in sync
        return item

    async def serialize(self):
        entries = [json.dumps(item) for item in self._items_list]
        with open(self.persistence_file, "w") as f:
            json.dump(entries, f)

    async def deserialize(self):
        with open(self.persistence_file, "r") as f:
            entries = json.load(f)
            for item in entries:
                await self.put(json.loads(item))  # Use put to keep the list in sync

task_queue = PersistentQueue()

@app.on_event("startup")
async def startup_event():
    global task_queue
    config = Config(".env")  # Load configuration from a .env file
    # make sure we only run 1 job at a time
    max_threads = config("MAX_THREADS", cast=int, default=1)
    await task_queue.deserialize()
    print(f'Index Service Starting, {task_queue.qsize()} jobs in queue')

@app.post("/submit_paper/")
async def submit_paper(paper: str, background_tasks: BackgroundTasks):
    global task_queue, job_id
    job_id = job_id + 1
    await task_queue.put({"id": job_id, "type": "dict", "paper": paper})
    await task_queue.serialize()
    return {"message": "Job submitted", "job_id": job_id}

@app.post("/submit_url/")
async def submit_url(url: str, background_tasks: BackgroundTasks):
    global task_queue, job_id
    job_id = job_id + 1
    await task_queue.put({"id": job_id, "type": "url", "url": url})
    await task_queue.serialize()
    return {"message": "Job submitted", "job_id": job_id}

@app.post("/submit_file/")
async def submit_url(filepath: str, background_tasks: BackgroundTasks):
    global task_queue, job_id
    job_id = job_id + 1
    await task_queue.put({"id": job_id, "type": "file", "filepath": filepath})
    await task_queue.serialize()
    return {"message": "Job submitted", "job_id": job_id}

async def process_tasks():
    global task_queue
    while True:
        spec = await task_queue.get()  # Wait for tasks
        print(f'Starting new task {spec}')
        try:
            if type(spec)  is str:
                spec = json.loads(spec)
                
            if spec['type'] == 'url':
                try:
                    await run_in_threadpool(s2.index_url, spec['url'])
                except:
                    print(f"fail to url file {spec}")
            if spec['type'] == 'dict':
                try:
                    paper_json = json.loads(spec['paper'])
                    await run_in_threadpool(s2.index_paper, paper_json)
                except:
                    print(f"fail to parse paper descriptor as json {spec['paper']}")
            if spec['type'] == 'file':
                try:
                    await run_in_threadpool(s2.index_file, spec['filepath'])
                except:
                    print(f"fail to index file {spec}")
        except Exception as e:
            traceback.print_exc()
        task_queue.task_done()
        await task_queue.serialize()

asyncio.create_task(process_tasks())

import OwlCoT as cot
cot = cot.OwlInnerVoice(None)
# set cot for rewrite so it can access llm
import semanticScholar2 as s2
s2.cot = cot
s2.rw.cot = cot


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=5006)
