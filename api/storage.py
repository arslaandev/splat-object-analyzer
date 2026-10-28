"""A durable FIFO queue for one GPU worker on one VM."""
from contextlib import contextmanager
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3


def now():
    return datetime.now(timezone.utc).isoformat(timespec='milliseconds').replace('+00:00', 'Z')


def write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, allow_nan=False))
    temporary.replace(path)


class QueueFull(Exception):
    pass


class Store:
    def __init__(self, directory, max_pending=8):
        self.root = Path(directory).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root/'jobs').mkdir(exist_ok=True)
        self.database = self.root/'jobs.sqlite3'
        self.max_pending = max_pending
        with self.connect() as db:
            db.execute('PRAGMA journal_mode=WAL')
            db.execute('CREATE TABLE IF NOT EXISTS jobs (sequence INTEGER PRIMARY KEY AUTOINCREMENT, id TEXT UNIQUE NOT NULL, status TEXT NOT NULL, record TEXT NOT NULL)')
            db.execute('CREATE INDEX IF NOT EXISTS jobs_status ON jobs(status, sequence)')

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.database, timeout=30)
        try:
            with db:
                yield db
        finally:
            db.close()

    def directory(self, job_id):
        return self.root/'jobs'/job_id

    def get(self, job_id):
        with self.connect() as db:
            row = db.execute('SELECT record FROM jobs WHERE id=?', (job_id,)).fetchone()
        return json.loads(row[0]) if row else None

    def create(self, record):
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            count = db.execute("SELECT COUNT(*) FROM jobs WHERE status IN ('queued','running')").fetchone()[0]
            if count >= self.max_pending:
                raise QueueFull()
            db.execute('INSERT INTO jobs(id,status,record) VALUES (?,?,?)', (record['id'], record['status'], json.dumps(record)))

    def pending_count(self):
        with self.connect() as db:
            return db.execute("SELECT COUNT(*) FROM jobs WHERE status IN ('queued','running')").fetchone()[0]

    def update(self, job_id, **changes):
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            row = db.execute('SELECT record FROM jobs WHERE id=?', (job_id,)).fetchone()
            record = json.loads(row[0])
            record.update(changes)
            db.execute('UPDATE jobs SET status=?,record=? WHERE id=?', (record['status'], json.dumps(record), job_id))
        return record

    def recover(self):
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            rows = db.execute("SELECT id,record FROM jobs WHERE status='running'").fetchall()
            for job_id, saved in rows:
                record = json.loads(saved)
                record.update(status='queued', progress=dict(stage='Resuming after service restart', completed=None, total=None))
                db.execute('UPDATE jobs SET status=?,record=? WHERE id=?', ('queued', json.dumps(record), job_id))

    def claim(self):
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            row = db.execute("SELECT id,record FROM jobs WHERE status='queued' ORDER BY sequence LIMIT 1").fetchone()
            if not row:
                return None
            job_id, value = row
            record = json.loads(value)
            record.update(status='running', started_at=now(), error=None,
                          progress=dict(stage='Starting '+record['kind'], completed=None, total=None))
            db.execute('UPDATE jobs SET status=?,record=? WHERE id=?', ('running', json.dumps(record), job_id))
        return record
