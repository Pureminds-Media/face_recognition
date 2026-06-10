"""Database layer for location-based visit tracking.

Supports SQLite (default, zero-config) and PostgreSQL (via DATABASE_URL).

Tables
------
locations   - camera-to-location mapping (camera_source -> human name)
sessions    - each server run (start/stop times)
visits      - continuous presence of a person at a location
"""

import os
import uuid
import sqlite3
import logging
import threading
from datetime import datetime, timezone, timedelta
from contextlib import contextmanager

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Backend detection
# ---------------------------------------------------------------------------
_backend = None          # "sqlite" | "postgres"
_sqlite_path = None      # only for sqlite
_pool = None             # only for postgres (psycopg2.pool.ThreadedConnectionPool)
_local = threading.local()  # thread-local sqlite connections

_DB_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_SQLITE = os.path.join(_DB_DIR, "face_recognition.db")

# ---------------------------------------------------------------------------
# Schema  (written to be compatible with both SQLite and PostgreSQL)
# ---------------------------------------------------------------------------

_SQLITE_SCHEMA = """
CREATE TABLE IF NOT EXISTS locations (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    camera_source   TEXT NOT NULL UNIQUE,
    name            TEXT NOT NULL,
    created_at      TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS sessions (
    id              TEXT PRIMARY KEY,
    started_at      TEXT NOT NULL,
    ended_at        TEXT,
    camera_source   TEXT
);

CREATE TABLE IF NOT EXISTS visits (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    person_name     TEXT NOT NULL,
    location_id     INTEGER REFERENCES locations(id),
    first_seen      TEXT NOT NULL,
    last_seen       TEXT NOT NULL,
    ended           INTEGER DEFAULT 0,
    confidence      REAL,
    session_id      TEXT REFERENCES sessions(id),
    screenshot      TEXT,
    footage         TEXT,
    visible_duration REAL,
    activity        TEXT,
    branch          TEXT NOT NULL DEFAULT 'Riyadh'
);

CREATE INDEX IF NOT EXISTS idx_visits_person     ON visits (person_name);
CREATE INDEX IF NOT EXISTS idx_visits_location   ON visits (location_id);
CREATE INDEX IF NOT EXISTS idx_visits_first_seen ON visits (first_seen);
CREATE INDEX IF NOT EXISTS idx_visits_open       ON visits (person_name, location_id) WHERE NOT ended;

CREATE TABLE IF NOT EXISTS people (
    name        TEXT PRIMARY KEY,
    section     TEXT NOT NULL DEFAULT '',
    branch      TEXT NOT NULL DEFAULT 'Riyadh',
    email       TEXT NOT NULL DEFAULT '',
    arabic_name TEXT NOT NULL DEFAULT '',
    shift       TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS sections (
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
    name    TEXT NOT NULL UNIQUE,
    manager TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS daily_reports (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    report_date     TEXT NOT NULL,
    generated_at    TEXT NOT NULL DEFAULT (datetime('now')),
    arrival_camera  TEXT NOT NULL DEFAULT '',
    exit_camera     TEXT NOT NULL DEFAULT '',
    work_start      TEXT NOT NULL DEFAULT '08:00',
    late_threshold  INTEGER NOT NULL DEFAULT 15,
    total_people    INTEGER NOT NULL DEFAULT 0,
    late_arrivals   INTEGER NOT NULL DEFAULT 0,
    late_exits      INTEGER NOT NULL DEFAULT 0,
    records_json    TEXT NOT NULL DEFAULT '[]',
    sent_email      INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_daily_reports_date ON daily_reports (report_date);

CREATE TABLE IF NOT EXISTS gate_events (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    person_name     TEXT NOT NULL,
    event_date      TEXT NOT NULL,
    exit_time       TEXT NOT NULL,
    entry_time      TEXT,
    duration_minutes REAL
);

CREATE INDEX IF NOT EXISTS idx_gate_events_person_date ON gate_events (person_name, event_date);
"""

_PG_SCHEMA = """
CREATE TABLE IF NOT EXISTS locations (
    id              SERIAL PRIMARY KEY,
    camera_source   TEXT NOT NULL UNIQUE,
    name            TEXT NOT NULL,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS sessions (
    id              UUID PRIMARY KEY,
    started_at      TIMESTAMPTZ NOT NULL,
    ended_at        TIMESTAMPTZ,
    camera_source   TEXT
);

CREATE TABLE IF NOT EXISTS visits (
    id              BIGSERIAL PRIMARY KEY,
    person_name     TEXT NOT NULL,
    location_id     INTEGER REFERENCES locations(id),
    first_seen      TIMESTAMPTZ NOT NULL,
    last_seen       TIMESTAMPTZ NOT NULL,
    ended           BOOLEAN DEFAULT FALSE,
    confidence      FLOAT,
    session_id      UUID REFERENCES sessions(id),
    screenshot      TEXT,
    footage         TEXT,
    visible_duration FLOAT,
    activity        TEXT,
    branch          TEXT NOT NULL DEFAULT 'Riyadh'
);

CREATE INDEX IF NOT EXISTS idx_visits_person     ON visits (person_name);
CREATE INDEX IF NOT EXISTS idx_visits_location   ON visits (location_id);
CREATE INDEX IF NOT EXISTS idx_visits_first_seen ON visits (first_seen);
CREATE INDEX IF NOT EXISTS idx_visits_open       ON visits (person_name, location_id) WHERE NOT ended;

CREATE TABLE IF NOT EXISTS people (
    name        TEXT PRIMARY KEY,
    section     TEXT NOT NULL DEFAULT '',
    branch      TEXT NOT NULL DEFAULT 'Riyadh',
    email       TEXT NOT NULL DEFAULT '',
    arabic_name TEXT NOT NULL DEFAULT '',
    shift       TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS sections (
    id      SERIAL PRIMARY KEY,
    name    TEXT NOT NULL UNIQUE,
    manager TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS daily_reports (
    id              SERIAL PRIMARY KEY,
    report_date     TEXT NOT NULL,
    generated_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    arrival_camera  TEXT NOT NULL DEFAULT '',
    exit_camera     TEXT NOT NULL DEFAULT '',
    work_start      TEXT NOT NULL DEFAULT '08:00',
    late_threshold  INTEGER NOT NULL DEFAULT 15,
    total_people    INTEGER NOT NULL DEFAULT 0,
    late_arrivals   INTEGER NOT NULL DEFAULT 0,
    late_exits      INTEGER NOT NULL DEFAULT 0,
    records_json    TEXT NOT NULL DEFAULT '[]',
    sent_email      BOOLEAN NOT NULL DEFAULT FALSE
);

CREATE INDEX IF NOT EXISTS idx_daily_reports_date ON daily_reports (report_date);

CREATE TABLE IF NOT EXISTS gate_events (
    id              SERIAL PRIMARY KEY,
    person_name     TEXT NOT NULL,
    event_date      TEXT NOT NULL,
    exit_time       TIMESTAMPTZ NOT NULL,
    entry_time      TIMESTAMPTZ,
    duration_minutes REAL
);

CREATE INDEX IF NOT EXISTS idx_gate_events_person_date ON gate_events (person_name, event_date);
"""


# ---------------------------------------------------------------------------
# Connection helpers
# ---------------------------------------------------------------------------

def init_db(dsn=None):
    """Initialise the database.

    - If *dsn* or DATABASE_URL starts with ``postgresql://`` / ``postgres://``,
      use PostgreSQL via psycopg2.
    - Otherwise fall back to SQLite at DATABASE_PATH (env) or
      ``face_recognition.db`` in the project directory.
    """
    global _backend, _pool, _sqlite_path

    dsn = dsn or os.getenv("DATABASE_URL", "")

    # --- Try PostgreSQL ---
    if dsn.startswith(("postgresql://", "postgres://")):
        try:
            import psycopg2
            import psycopg2.pool
            import psycopg2.extras
            _pool = psycopg2.pool.ThreadedConnectionPool(1, 5, dsn)
            with _pg_cursor(commit=True) as cur:
                cur.execute(_PG_SCHEMA)
                # Migration: add screenshot column to existing databases
                try:
                    cur.execute("ALTER TABLE visits ADD COLUMN screenshot TEXT")
                except Exception:
                    pass  # column already exists
                # Migration: add footage column to existing databases
                try:
                    cur.execute("ALTER TABLE visits ADD COLUMN footage TEXT")
                except Exception:
                    pass  # column already exists
                # Migration: add visible_duration column to existing databases
                try:
                    cur.execute("ALTER TABLE visits ADD COLUMN visible_duration FLOAT")
                except Exception:
                    pass  # column already exists
                # Migration: add activity column to existing databases
                try:
                    cur.execute("ALTER TABLE visits ADD COLUMN activity TEXT")
                except Exception:
                    pass  # column already exists
                # Migration: add branch column to existing databases
                try:
                    cur.execute("ALTER TABLE visits ADD COLUMN branch TEXT DEFAULT 'Riyadh'")
                except Exception:
                    pass  # column already exists
                try:
                    cur.execute("UPDATE visits SET branch = 'Riyadh' WHERE branch IS NULL")
                except Exception:
                    pass
                try:
                    cur.execute("CREATE TABLE IF NOT EXISTS people (name TEXT PRIMARY KEY, section TEXT NOT NULL DEFAULT '', branch TEXT NOT NULL DEFAULT 'Riyadh', email TEXT NOT NULL DEFAULT '')")
                except Exception:
                    pass
                try:
                    cur.execute("ALTER TABLE people ADD COLUMN email TEXT NOT NULL DEFAULT ''")
                except Exception:
                    pass
                try:
                    cur.execute("CREATE TABLE IF NOT EXISTS sections (id SERIAL PRIMARY KEY, name TEXT NOT NULL UNIQUE, manager TEXT NOT NULL DEFAULT '')")
                except Exception:
                    pass
                try:
                    cur.execute("ALTER TABLE sections ADD COLUMN manager TEXT NOT NULL DEFAULT ''")
                except Exception:
                    pass
                try:
                    cur.execute("ALTER TABLE people ADD COLUMN arabic_name TEXT NOT NULL DEFAULT ''")
                except Exception:
                    pass
                try:
                    cur.execute("ALTER TABLE people ADD COLUMN shift TEXT NOT NULL DEFAULT ''")
                except Exception:
                    pass
                try:
                    cur.execute("""CREATE TABLE IF NOT EXISTS daily_reports (
                        id SERIAL PRIMARY KEY,
                        report_date TEXT NOT NULL,
                        generated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        arrival_camera TEXT NOT NULL DEFAULT '',
                        exit_camera TEXT NOT NULL DEFAULT '',
                        work_start TEXT NOT NULL DEFAULT '08:00',
                        late_threshold INTEGER NOT NULL DEFAULT 15,
                        total_people INTEGER NOT NULL DEFAULT 0,
                        late_arrivals INTEGER NOT NULL DEFAULT 0,
                        late_exits INTEGER NOT NULL DEFAULT 0,
                        records_json TEXT NOT NULL DEFAULT '[]',
                        sent_email BOOLEAN NOT NULL DEFAULT FALSE
                    )""")
                    cur.execute("CREATE INDEX IF NOT EXISTS idx_daily_reports_date ON daily_reports (report_date)")
                except Exception:
                    pass
                try:
                    cur.execute("""CREATE TABLE IF NOT EXISTS gate_events (
                        id SERIAL PRIMARY KEY,
                        person_name TEXT NOT NULL,
                        event_date TEXT NOT NULL,
                        exit_time TIMESTAMPTZ NOT NULL,
                        entry_time TIMESTAMPTZ,
                        duration_minutes REAL
                    )""")
                    cur.execute("CREATE INDEX IF NOT EXISTS idx_gate_events_person_date ON gate_events (person_name, event_date)")
                except Exception:
                    pass
                try:
                    cur.execute("""CREATE TABLE IF NOT EXISTS zones (
                        id SERIAL PRIMARY KEY,
                        name TEXT NOT NULL UNIQUE,
                        description TEXT NOT NULL DEFAULT '',
                        branch TEXT NOT NULL DEFAULT 'Riyadh',
                        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                    )""")
                    cur.execute("""CREATE TABLE IF NOT EXISTS zone_cameras (
                        zone_id INTEGER NOT NULL REFERENCES zones(id) ON DELETE CASCADE,
                        location_id INTEGER NOT NULL REFERENCES locations(id) ON DELETE CASCADE,
                        PRIMARY KEY (zone_id, location_id)
                    )""")
                except Exception:
                    pass
                try:
                    cur.execute("ALTER TABLE people ADD COLUMN home_zone_id INTEGER REFERENCES zones(id)")
                except Exception:
                    pass
            _backend = "postgres"
            log.info("Database initialised (PostgreSQL)")
            return
        except Exception as e:
            log.warning("PostgreSQL connection failed (%s) – falling back to SQLite", e)
            _pool = None

    # --- SQLite fallback ---
    _sqlite_path = os.getenv("DATABASE_PATH", _DEFAULT_SQLITE)
    try:
        conn = _sqlite_get_conn()
        conn.executescript(_SQLITE_SCHEMA)
        # Migration: add screenshot column to existing databases
        try:
            conn.execute("ALTER TABLE visits ADD COLUMN screenshot TEXT")
            conn.commit()
        except Exception:
            pass  # column already exists
        # Migration: add footage column to existing databases
        try:
            conn.execute("ALTER TABLE visits ADD COLUMN footage TEXT")
            conn.commit()
        except Exception:
            pass  # column already exists
        # Migration: add visible_duration column to existing databases
        try:
            conn.execute("ALTER TABLE visits ADD COLUMN visible_duration REAL")
            conn.commit()
        except Exception:
            pass  # column already exists
        # Migration: add activity column to existing databases
        try:
            conn.execute("ALTER TABLE visits ADD COLUMN activity TEXT")
            conn.commit()
        except Exception:
            pass  # column already exists
        # Migration: add branch column to existing databases
        try:
            conn.execute("ALTER TABLE visits ADD COLUMN branch TEXT DEFAULT 'Riyadh'")
            conn.commit()
        except Exception:
            pass  # column already exists
        try:
            conn.execute("UPDATE visits SET branch = 'Riyadh' WHERE branch IS NULL")
            conn.commit()
        except Exception:
            pass
        try:
            conn.execute("CREATE TABLE IF NOT EXISTS people (name TEXT PRIMARY KEY, section TEXT NOT NULL DEFAULT '', branch TEXT NOT NULL DEFAULT 'Riyadh', email TEXT NOT NULL DEFAULT '')")
            conn.commit()
        except Exception:
            pass
        try:
            conn.execute("ALTER TABLE people ADD COLUMN email TEXT NOT NULL DEFAULT ''")
            conn.commit()
        except Exception:
            pass
        try:
            conn.execute("CREATE TABLE IF NOT EXISTS sections (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL UNIQUE, manager TEXT NOT NULL DEFAULT '')")
            conn.commit()
        except Exception:
            pass
        try:
            conn.execute("ALTER TABLE sections ADD COLUMN manager TEXT NOT NULL DEFAULT ''")
            conn.commit()
        except Exception:
            pass
        try:
            conn.execute("ALTER TABLE people ADD COLUMN arabic_name TEXT NOT NULL DEFAULT ''")
            conn.commit()
        except Exception:
            pass
        try:
            conn.execute("ALTER TABLE people ADD COLUMN shift TEXT NOT NULL DEFAULT ''")
            conn.commit()
        except Exception:
            pass
        try:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS daily_reports (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    report_date     TEXT NOT NULL,
                    generated_at    TEXT NOT NULL DEFAULT (datetime('now')),
                    arrival_camera  TEXT NOT NULL DEFAULT '',
                    exit_camera     TEXT NOT NULL DEFAULT '',
                    work_start      TEXT NOT NULL DEFAULT '08:00',
                    late_threshold  INTEGER NOT NULL DEFAULT 15,
                    total_people    INTEGER NOT NULL DEFAULT 0,
                    late_arrivals   INTEGER NOT NULL DEFAULT 0,
                    late_exits      INTEGER NOT NULL DEFAULT 0,
                    records_json    TEXT NOT NULL DEFAULT '[]',
                    sent_email      INTEGER NOT NULL DEFAULT 0
                );
                CREATE INDEX IF NOT EXISTS idx_daily_reports_date ON daily_reports (report_date);
            """)
            conn.commit()
        except Exception:
            pass
        try:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS gate_events (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    person_name     TEXT NOT NULL,
                    event_date      TEXT NOT NULL,
                    exit_time       TEXT NOT NULL,
                    entry_time      TEXT,
                    duration_minutes REAL
                );
                CREATE INDEX IF NOT EXISTS idx_gate_events_person_date ON gate_events (person_name, event_date);
            """)
            conn.commit()
        except Exception:
            pass
        try:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS zones (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    name        TEXT NOT NULL UNIQUE,
                    description TEXT NOT NULL DEFAULT '',
                    branch      TEXT NOT NULL DEFAULT 'Riyadh',
                    created_at  TEXT DEFAULT (datetime('now'))
                );
                CREATE TABLE IF NOT EXISTS zone_cameras (
                    zone_id     INTEGER NOT NULL REFERENCES zones(id) ON DELETE CASCADE,
                    location_id INTEGER NOT NULL REFERENCES locations(id) ON DELETE CASCADE,
                    PRIMARY KEY (zone_id, location_id)
                );
            """)
            conn.commit()
        except Exception:
            pass
        try:
            conn.execute("ALTER TABLE people ADD COLUMN home_zone_id INTEGER REFERENCES zones(id)")
            conn.commit()
        except Exception:
            pass
        conn.commit()
        _backend = "sqlite"
        log.info("Database initialised (SQLite: %s)", _sqlite_path)
    except Exception as e:
        log.error("Failed to initialise SQLite database: %s", e)
        _backend = None


def close_db():
    """Close all connections."""
    global _backend, _pool
    if _backend == "postgres" and _pool is not None:
        _pool.closeall()
        _pool = None
    # SQLite connections are per-thread; they will be closed when threads end.
    # Close the current thread's connection if it exists.
    conn = getattr(_local, "conn", None)
    if conn is not None:
        try:
            conn.close()
        except Exception:
            pass
        _local.conn = None
    _backend = None


def is_available():
    """Return True if the DB backend is initialised."""
    return _backend is not None


# ---------------------------------------------------------------------------
# Internal cursor helpers
# ---------------------------------------------------------------------------

def _sqlite_get_conn():
    """Get or create a thread-local SQLite connection."""
    conn = getattr(_local, "conn", None)
    if conn is None:
        assert _sqlite_path is not None, "SQLite path not configured"
        conn = sqlite3.connect(_sqlite_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        # Auto-checkpoint every ~1000 pages (~4 MB) so the WAL file doesn't
        # balloon during long uptime and stall reads on flush.
        conn.execute("PRAGMA wal_autocheckpoint=1000")
        conn.execute("PRAGMA foreign_keys=ON")
        _local.conn = conn
    return conn


@contextmanager
def _sqlite_cursor(commit=False):
    """Yield a sqlite3 cursor, optionally committing on success."""
    conn = _sqlite_get_conn()
    cur = conn.cursor()
    try:
        yield cur
        if commit:
            conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        cur.close()


@contextmanager
def _pg_cursor(commit=False):
    """Yield a PostgreSQL RealDictCursor from the pool."""
    import psycopg2.extras
    assert _pool is not None, "PostgreSQL pool not initialised"
    conn = _pool.getconn()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            yield cur
            if commit:
                conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        _pool.putconn(conn)


@contextmanager
def _cursor(commit=False):
    """Backend-agnostic cursor context manager."""
    if _backend == "postgres":
        with _pg_cursor(commit=commit) as cur:
            yield cur
    else:
        with _sqlite_cursor(commit=commit) as cur:
            yield cur


def _now():
    return datetime.now(timezone.utc)


def _now_str():
    """Return current UTC time as ISO string (for SQLite)."""
    return _now().isoformat()


def _param(sql):
    """Convert %s placeholders to ? for SQLite."""
    if _backend == "sqlite":
        return sql.replace("%s", "?")
    return sql


def _row_to_dict(row):
    """Convert a sqlite3.Row or psycopg2 RealDictRow to a plain dict."""
    if row is None:
        return None
    if isinstance(row, dict):
        return row
    # sqlite3.Row
    return dict(row)


def _rows_to_dicts(rows):
    """Convert a list of rows to a list of dicts."""
    return [_row_to_dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Locations
# ---------------------------------------------------------------------------

def upsert_location(camera_source, name):
    """Insert or update a location. Returns the location id."""
    with _cursor(commit=True) as cur:
        if _backend == "postgres":
            cur.execute(
                """
                INSERT INTO locations (camera_source, name)
                VALUES (%s, %s)
                ON CONFLICT (camera_source) DO UPDATE SET name = EXCLUDED.name
                RETURNING id
                """,
                (str(camera_source), str(name)),
            )
            return cur.fetchone()["id"]
        else:
            # SQLite: INSERT OR REPLACE would reset id, so use upsert
            cur.execute(
                """
                INSERT INTO locations (camera_source, name)
                VALUES (?, ?)
                ON CONFLICT (camera_source) DO UPDATE SET name = EXCLUDED.name
                """,
                (str(camera_source), str(name)),
            )
            cur.execute(
                "SELECT id FROM locations WHERE camera_source = ?",
                (str(camera_source),),
            )
            return cur.fetchone()[0]


def get_location_by_source(camera_source):
    """Return location row for a camera_source, or None."""
    sql = _param("SELECT id, camera_source, name FROM locations WHERE camera_source = %s")
    with _cursor() as cur:
        cur.execute(sql, (str(camera_source),))
        return _row_to_dict(cur.fetchone())


def get_locations():
    """Return all locations."""
    with _cursor() as cur:
        cur.execute("SELECT id, camera_source, name FROM locations ORDER BY id")
        return _rows_to_dicts(cur.fetchall())


# ---------------------------------------------------------------------------
# Sessions
# ---------------------------------------------------------------------------

def create_session(camera_source=None):
    """Start a new session. Returns session UUID string."""
    sid = str(uuid.uuid4())
    now = _now_str() if _backend == "sqlite" else _now()
    sql = _param("INSERT INTO sessions (id, started_at, camera_source) VALUES (%s, %s, %s)")
    with _cursor(commit=True) as cur:
        cur.execute(sql, (sid, now, str(camera_source) if camera_source else None))
    return sid


def end_session(session_id):
    """Mark a session as ended."""
    if not session_id:
        return
    now = _now_str() if _backend == "sqlite" else _now()
    sql = _param("UPDATE sessions SET ended_at = %s WHERE id = %s AND ended_at IS NULL")
    with _cursor(commit=True) as cur:
        cur.execute(sql, (now, session_id))


def get_sessions(limit=50, offset=0):
    """Return recent sessions."""
    sql = _param("""
        SELECT id, started_at, ended_at, camera_source
        FROM sessions ORDER BY started_at DESC LIMIT %s OFFSET %s
    """)
    with _cursor() as cur:
        cur.execute(sql, (limit, offset))
        return _rows_to_dicts(cur.fetchall())


# ---------------------------------------------------------------------------
# Visits
# ---------------------------------------------------------------------------

def open_visit(person_name, location_id, confidence=None, session_id=None, branch='Riyadh'):
    """Create a new open visit. Returns the visit id."""
    now = _now_str() if _backend == "sqlite" else _now()
    ended_val = 0 if _backend == "sqlite" else False
    with _cursor(commit=True) as cur:
        if _backend == "postgres":
            cur.execute(
                """
                INSERT INTO visits (person_name, location_id, first_seen, last_seen,
                                    ended, confidence, session_id, branch)
                VALUES (%s, %s, %s, %s, FALSE, %s, %s, %s)
                RETURNING id
                """,
                (person_name, location_id, now, now, confidence, session_id, branch),
            )
            return cur.fetchone()["id"]
        else:
            cur.execute(
                """
                INSERT INTO visits (person_name, location_id, first_seen, last_seen,
                                    ended, confidence, session_id, branch)
                VALUES (?, ?, ?, ?, 0, ?, ?, ?)
                """,
                (person_name, location_id, now, now, confidence, session_id, branch),
            )
            return cur.lastrowid


def update_visit_seen(visit_id, confidence=None):
    """Bump last_seen and optionally update confidence (keep best)."""
    now = _now_str() if _backend == "sqlite" else _now()
    with _cursor(commit=True) as cur:
        if confidence is not None:
            sql = _param("""
                UPDATE visits SET last_seen = %s,
                    confidence = MIN(confidence, %s)
                WHERE id = %s AND NOT ended
            """)
            # PostgreSQL uses LEAST(), SQLite uses MIN()
            if _backend == "postgres":
                sql = """
                    UPDATE visits SET last_seen = %s,
                        confidence = LEAST(confidence, %s)
                    WHERE id = %s AND NOT ended
                """
            cur.execute(sql, (now, confidence, visit_id))
        else:
            sql = _param("UPDATE visits SET last_seen = %s WHERE id = %s AND NOT ended")
            cur.execute(sql, (now, visit_id))


def close_visit(visit_id):
    """Mark a visit as ended."""
    now = _now_str() if _backend == "sqlite" else _now()
    if _backend == "postgres":
        sql = "UPDATE visits SET ended = TRUE, last_seen = %s WHERE id = %s AND NOT ended"
    else:
        sql = "UPDATE visits SET ended = 1, last_seen = ? WHERE id = ? AND NOT ended"
    with _cursor(commit=True) as cur:
        cur.execute(sql, (now, visit_id))


def update_visit_screenshot(visit_id, screenshot):
    """Set the screenshot filename for a visit."""
    sql = _param("UPDATE visits SET screenshot = %s WHERE id = %s")
    with _cursor(commit=True) as cur:
        cur.execute(sql, (screenshot, visit_id))


def update_visit_footage(visit_id, footage):
    """Set the footage filename for a visit."""
    sql = _param("UPDATE visits SET footage = %s WHERE id = %s")
    with _cursor(commit=True) as cur:
        cur.execute(sql, (footage, visit_id))


def update_visit_visible_duration(visit_id, visible_duration):
    """Set the visible duration (seconds on camera) for a visit."""
    sql = _param("UPDATE visits SET visible_duration = %s WHERE id = %s")
    with _cursor(commit=True) as cur:
        cur.execute(sql, (visible_duration, visit_id))


def update_visit_activity(visit_id, activity):
    """Set the most frequent activity label for a visit."""
    sql = _param("UPDATE visits SET activity = %s WHERE id = %s")
    with _cursor(commit=True) as cur:
        cur.execute(sql, (activity, visit_id))


def get_open_visit(person_name, location_id):
    """Return the open visit for a person at a location, or None."""
    sql = _param("""
        SELECT id, person_name, location_id, first_seen, last_seen, confidence, screenshot, footage, visible_duration, activity
        FROM visits
        WHERE person_name = %s AND location_id = %s AND NOT ended
        ORDER BY first_seen DESC LIMIT 1
    """)
    with _cursor() as cur:
        cur.execute(sql, (person_name, location_id))
        return _row_to_dict(cur.fetchone())


def get_all_open_visits():
    """Return all open visits (for stale-check)."""
    with _cursor() as cur:
        cur.execute("""
            SELECT v.id, v.person_name, v.location_id, v.first_seen, v.last_seen,
                   v.confidence, v.screenshot, v.footage, v.visible_duration, v.activity,
                   l.name as location_name, l.camera_source
            FROM visits v
            JOIN locations l ON l.id = v.location_id
            WHERE NOT v.ended
            ORDER BY v.last_seen DESC
        """)
        return _rows_to_dicts(cur.fetchall())


def close_stale_visits(timeout_minutes=10):
    """Close visits where last_seen is older than timeout."""
    if _backend == "postgres":
        cutoff = _now() - timedelta(minutes=timeout_minutes)
        sql = "UPDATE visits SET ended = TRUE WHERE NOT ended AND last_seen < %s"
    else:
        cutoff = (_now() - timedelta(minutes=timeout_minutes)).isoformat()
        sql = "UPDATE visits SET ended = 1 WHERE NOT ended AND last_seen < ?"
    with _cursor(commit=True) as cur:
        cur.execute(sql, (cutoff,))
        return cur.rowcount


def close_all_open_visits():
    """Close every open visit (used on shutdown)."""
    if _backend == "postgres":
        sql = "UPDATE visits SET ended = TRUE WHERE NOT ended"
    else:
        sql = "UPDATE visits SET ended = 1 WHERE NOT ended"
    with _cursor(commit=True) as cur:
        cur.execute(sql)
        return cur.rowcount


# ---------------------------------------------------------------------------
# Queries for reports
# ---------------------------------------------------------------------------

def get_person_visits(person_name, date_from=None, date_to=None, limit=200, branch=None):
    """All visits for a person, optionally filtered by date range and branch."""
    ph = "?" if _backend == "sqlite" else "%s"
    clauses = [f"v.person_name = {ph}"]
    params = [person_name]
    if date_from:
        clauses.append(f"v.first_seen >= {ph}")
        params.append(date_from if _backend == "postgres" else str(date_from))
    if date_to:
        clauses.append(f"v.first_seen < {ph}")
        params.append(date_to if _backend == "postgres" else str(date_to))
    if branch:
        clauses.append(f"v.branch = {ph}")
        params.append(branch)
    params.append(limit)

    sql = f"""
        SELECT v.id, v.person_name, v.first_seen, v.last_seen, v.ended,
               v.confidence, v.screenshot, v.footage, v.visible_duration, v.activity,
               l.name as location_name, l.camera_source
        FROM visits v
        JOIN locations l ON l.id = v.location_id
        WHERE {' AND '.join(clauses)}
        ORDER BY v.first_seen DESC
        LIMIT {ph}
    """
    with _cursor() as cur:
        cur.execute(sql, params)
        return _rows_to_dicts(cur.fetchall())


def get_location_visits(location_id, date_from=None, date_to=None, limit=200, branch=None):
    """All visits at a location, optionally filtered by date range and branch."""
    ph = "?" if _backend == "sqlite" else "%s"
    clauses = [f"v.location_id = {ph}"]
    params = [location_id]
    if date_from:
        clauses.append(f"v.first_seen >= {ph}")
        params.append(date_from if _backend == "postgres" else str(date_from))
    if date_to:
        clauses.append(f"v.first_seen < {ph}")
        params.append(date_to if _backend == "postgres" else str(date_to))
    if branch:
        clauses.append(f"v.branch = {ph}")
        params.append(branch)
    params.append(limit)

    sql = f"""
        SELECT v.id, v.person_name, v.first_seen, v.last_seen, v.ended,
               v.confidence, v.screenshot, v.footage, v.visible_duration, v.activity,
               l.name as location_name
        FROM visits v
        JOIN locations l ON l.id = v.location_id
        WHERE {' AND '.join(clauses)}
        ORDER BY v.first_seen DESC
        LIMIT {ph}
    """
    with _cursor() as cur:
        cur.execute(sql, params)
        return _rows_to_dicts(cur.fetchall())


def get_daily_summary(date, branch=None):
    """All visits for a calendar day, grouped by person then time.

    Parameters
    ----------
    date : datetime.date
        The calendar day to query.
    branch : str or None
        When provided, filter to this branch only.

    Returns a list of dicts sorted by first_seen descending (most recent first).
    """
    day_start = datetime(date.year, date.month, date.day, tzinfo=timezone.utc)
    day_end = day_start + timedelta(days=1)

    ph = "?" if _backend == "sqlite" else "%s"
    if _backend == "postgres":
        params = [day_start, day_end]
    else:
        params = [day_start.isoformat(), day_end.isoformat()]

    branch_clause = ""
    if branch:
        branch_clause = f"AND v.branch = {ph}"
        params.append(branch)

    sql = f"""
        SELECT v.id, v.person_name, v.first_seen, v.last_seen, v.ended,
               v.confidence, v.screenshot, v.footage, v.visible_duration, v.activity,
               l.name as location_name, l.camera_source
        FROM visits v
        JOIN locations l ON l.id = v.location_id
        WHERE v.first_seen >= {ph} AND v.first_seen < {ph}
        {branch_clause}
        ORDER BY v.first_seen DESC
    """
    with _cursor() as cur:
        cur.execute(sql, params)
        return _rows_to_dicts(cur.fetchall())


def get_known_persons(limit=500, branch=None):
    """Return distinct person names that have visits, optionally filtered by branch."""
    ph = "?" if _backend == "sqlite" else "%s"
    if branch:
        sql = f"SELECT DISTINCT person_name FROM visits WHERE branch = {ph} ORDER BY person_name LIMIT {ph}"
        params = (branch, limit)
    else:
        sql = f"SELECT DISTINCT person_name FROM visits ORDER BY person_name LIMIT {ph}"
        params = (limit,)
    with _cursor() as cur:
        cur.execute(sql, params)
        return [dict(r)["person_name"] for r in cur.fetchall()]


def rename_person(old_name, new_name):
    """Rename a person across all visits. Returns number of rows updated."""
    ph = "?" if _backend == "sqlite" else "%s"
    sql = _param(f"UPDATE visits SET person_name = {ph} WHERE person_name = {ph}")
    with _cursor(commit=True) as cur:
        cur.execute(sql, (new_name, old_name))
        return cur.rowcount


def delete_person_visits(person_name):
    """Delete all visits for a person. Returns number of rows deleted."""
    ph = "?" if _backend == "sqlite" else "%s"
    sql = _param(f"DELETE FROM visits WHERE person_name = {ph}")
    with _cursor(commit=True) as cur:
        cur.execute(sql, (person_name,))
        return cur.rowcount


def clear_all_data():
    """Delete all visits and sessions. Locations are kept (they map to cameras).

    Returns the number of visits deleted.
    """
    with _cursor(commit=True) as cur:
        cur.execute("DELETE FROM visits")
        visit_count = cur.rowcount
        cur.execute("DELETE FROM sessions")
    return visit_count


# ---------------------------------------------------------------------------
# People metadata
# ---------------------------------------------------------------------------

def get_person_meta(name):
    """Return {name, section, branch, email, arabic_name, shift} for a person, or None if not found."""
    ph = "?" if _backend == "sqlite" else "%s"
    with _cursor() as cur:
        cur.execute(f"SELECT name, section, branch, email, arabic_name, shift FROM people WHERE name = {ph}", (name,))
        row = cur.fetchone()
    return _row_to_dict(row) if row else None


def upsert_person_meta(name, section=None, branch=None, email=None, arabic_name=None, shift=None):
    """Insert or update section/branch/email/arabic_name/shift for a person."""
    ph = "?" if _backend == "sqlite" else "%s"
    existing = get_person_meta(name)
    if existing is None:
        section = section if section is not None else ""
        branch = branch if branch is not None else "Riyadh"
        email = email if email is not None else ""
        arabic_name = arabic_name if arabic_name is not None else ""
        shift = shift if shift is not None else ""
        with _cursor(commit=True) as cur:
            cur.execute(
                f"INSERT INTO people (name, section, branch, email, arabic_name, shift) VALUES ({ph},{ph},{ph},{ph},{ph},{ph})",
                (name, section, branch, email, arabic_name, shift),
            )
    else:
        updates, params = [], []
        if section is not None:
            updates.append(f"section = {ph}"); params.append(section)
        if branch is not None:
            updates.append(f"branch = {ph}"); params.append(branch)
        if email is not None:
            updates.append(f"email = {ph}"); params.append(email)
        if arabic_name is not None:
            updates.append(f"arabic_name = {ph}"); params.append(arabic_name)
        if shift is not None:
            updates.append(f"shift = {ph}"); params.append(shift)
        if updates:
            params.append(name)
            with _cursor(commit=True) as cur:
                cur.execute(f"UPDATE people SET {', '.join(updates)} WHERE name = {ph}", params)


def rename_person_meta(old_name, new_name):
    """Rename a person row in the people table."""
    ph = "?" if _backend == "sqlite" else "%s"
    with _cursor(commit=True) as cur:
        cur.execute(f"UPDATE people SET name = {ph} WHERE name = {ph}", (new_name, old_name))


def delete_person_meta(name):
    """Remove a person's metadata row."""
    ph = "?" if _backend == "sqlite" else "%s"
    with _cursor(commit=True) as cur:
        cur.execute(f"DELETE FROM people WHERE name = {ph}", (name,))


def get_all_people_meta():
    """Return list of {name, section, branch, email, arabic_name, shift, home_zone_id} for all people with metadata."""
    with _cursor() as cur:
        cur.execute("SELECT name, section, branch, email, arabic_name, shift, home_zone_id FROM people ORDER BY name")
        return _rows_to_dicts(cur.fetchall())


def get_branch_members(branch):
    """Return list of person names assigned to a branch."""
    ph = "?" if _backend == "sqlite" else "%s"
    with _cursor() as cur:
        cur.execute(f"SELECT name FROM people WHERE branch = {ph} ORDER BY name", (branch,))
        return [r["name"] for r in _rows_to_dicts(cur.fetchall())]


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------

def get_all_sections():
    """Return list of {id, name, manager} for all sections."""
    with _cursor() as cur:
        cur.execute("SELECT id, name, manager FROM sections ORDER BY name")
        return _rows_to_dicts(cur.fetchall())


def set_section_manager(section_name, person_name):
    """Set or clear the manager for a section. Pass '' to clear."""
    ph = "?" if _backend == "sqlite" else "%s"
    with _cursor(commit=True) as cur:
        cur.execute(f"UPDATE sections SET manager = {ph} WHERE name = {ph}", (person_name, section_name))


def create_section(name):
    """Insert a new section. Returns {id, name} or raises on duplicate."""
    ph = "?" if _backend == "sqlite" else "%s"
    if _backend == "postgres":
        with _cursor(commit=True) as cur:
            cur.execute(
                "INSERT INTO sections (name) VALUES (%s) RETURNING id, name",
                (name,),
            )
            return _row_to_dict(cur.fetchone())
    else:
        with _cursor(commit=True) as cur:
            cur.execute("INSERT INTO sections (name) VALUES (?)", (name,))
            new_id = cur.lastrowid
        return {"id": new_id, "name": name}


def rename_section(old_name, new_name):
    """Rename a section and update all people assigned to it."""
    ph = "?" if _backend == "sqlite" else "%s"
    with _cursor(commit=True) as cur:
        cur.execute(f"UPDATE sections SET name = {ph} WHERE name = {ph}", (new_name, old_name))
        cur.execute(f"UPDATE people SET section = {ph} WHERE section = {ph}", (new_name, old_name))


def delete_section(name):
    """Delete a section and clear people.section for anyone in it."""
    ph = "?" if _backend == "sqlite" else "%s"
    with _cursor(commit=True) as cur:
        cur.execute(f"UPDATE people SET section = '' WHERE section = {ph}", (name,))
        cur.execute(f"DELETE FROM sections WHERE name = {ph}", (name,))


def get_section_members(section_name):
    """Return list of person names assigned to a section."""
    ph = "?" if _backend == "sqlite" else "%s"
    with _cursor() as cur:
        cur.execute(f"SELECT name FROM people WHERE section = {ph} ORDER BY name", (section_name,))
        return [r["name"] for r in _rows_to_dicts(cur.fetchall())]


def assign_person_section(person_name, section_name):
    """Set people.section = section_name for a person (upsert if missing)."""
    ph = "?" if _backend == "sqlite" else "%s"
    existing = get_person_meta(person_name)
    if existing is None:
        with _cursor(commit=True) as cur:
            cur.execute(
                f"INSERT INTO people (name, section, branch, email) VALUES ({ph},{ph},{ph},{ph})",
                (person_name, section_name, "Riyadh", ""),
            )
    else:
        with _cursor(commit=True) as cur:
            cur.execute(
                f"UPDATE people SET section = {ph} WHERE name = {ph}",
                (section_name, person_name),
            )


def remove_person_section(person_name):
    """Clear people.section for a person."""
    ph = "?" if _backend == "sqlite" else "%s"
    with _cursor(commit=True) as cur:
        cur.execute(f"UPDATE people SET section = '' WHERE name = {ph}", (person_name,))


# ---------------------------------------------------------------------------
# Daily reports
# ---------------------------------------------------------------------------

def save_daily_report(report_date, arrival_camera, exit_camera, work_start,
                      late_threshold, records, sent_email=False):
    """Insert or replace today's report snapshot. Returns the row id."""
    import json as _json
    ph = "?" if _backend == "sqlite" else "%s"
    records_json = _json.dumps(records, ensure_ascii=False, default=str)
    total = len(records)
    late_arr = sum(1 for r in records if r.get("arrived_late"))
    late_ext = sum(1 for r in records if r.get("late_exits_count", 0) > 0)
    sent = 1 if sent_email else 0
    now_str = datetime.now(timezone.utc).isoformat()

    # Upsert: replace existing row for the same date if present
    with _cursor(commit=True) as cur:
        if _backend == "sqlite":
            cur.execute("""
                INSERT INTO daily_reports
                    (report_date, generated_at, arrival_camera, exit_camera,
                     work_start, late_threshold, total_people, late_arrivals,
                     late_exits, records_json, sent_email)
                VALUES (?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT DO NOTHING
            """, (report_date, now_str, arrival_camera, exit_camera,
                  work_start, late_threshold, total, late_arr, late_ext,
                  records_json, sent))
            # Always update so re-running overwrites with fresh data
            cur.execute("""
                UPDATE daily_reports SET
                    generated_at=?, arrival_camera=?, exit_camera=?,
                    work_start=?, late_threshold=?, total_people=?,
                    late_arrivals=?, late_exits=?, records_json=?, sent_email=?
                WHERE report_date=?
            """, (now_str, arrival_camera, exit_camera, work_start,
                  late_threshold, total, late_arr, late_ext,
                  records_json, sent, report_date))
        else:
            cur.execute(f"""
                INSERT INTO daily_reports
                    (report_date, generated_at, arrival_camera, exit_camera,
                     work_start, late_threshold, total_people, late_arrivals,
                     late_exits, records_json, sent_email)
                VALUES ({ph},{ph},{ph},{ph},{ph},{ph},{ph},{ph},{ph},{ph},{ph})
                ON CONFLICT (report_date) DO UPDATE SET
                    generated_at=EXCLUDED.generated_at,
                    arrival_camera=EXCLUDED.arrival_camera,
                    exit_camera=EXCLUDED.exit_camera,
                    work_start=EXCLUDED.work_start,
                    late_threshold=EXCLUDED.late_threshold,
                    total_people=EXCLUDED.total_people,
                    late_arrivals=EXCLUDED.late_arrivals,
                    late_exits=EXCLUDED.late_exits,
                    records_json=EXCLUDED.records_json,
                    sent_email=EXCLUDED.sent_email
            """, (report_date, now_str, arrival_camera, exit_camera,
                  work_start, late_threshold, total, late_arr, late_ext,
                  records_json, sent))


def get_daily_report(report_date):
    """Return a single saved report row or None."""
    import json as _json
    ph = "?" if _backend == "sqlite" else "%s"
    with _cursor() as cur:
        cur.execute(
            f"SELECT * FROM daily_reports WHERE report_date = {ph}", (report_date,)
        )
        row = _row_to_dict(cur.fetchone())
    if row and row.get("records_json"):
        try:
            row["records"] = _json.loads(row["records_json"])
        except Exception:
            row["records"] = []
    return row


def list_daily_reports(limit=60):
    """Return most recent saved report summaries (no records_json)."""
    with _cursor() as cur:
        cur.execute("""
            SELECT id, report_date, generated_at, arrival_camera, exit_camera,
                   work_start, late_threshold, total_people, late_arrivals,
                   late_exits, sent_email
            FROM daily_reports
            ORDER BY report_date DESC
            LIMIT ?
        """ if _backend == "sqlite" else """
            SELECT id, report_date, generated_at, arrival_camera, exit_camera,
                   work_start, late_threshold, total_people, late_arrivals,
                   late_exits, sent_email
            FROM daily_reports
            ORDER BY report_date DESC
            LIMIT %s
        """, (limit,))
        return _rows_to_dicts(cur.fetchall())


# ---------------------------------------------------------------------------
# Gate events (real-time exit/entry tracking)
# ---------------------------------------------------------------------------

def open_gate_exit(person_name, exit_dt):
    """Open a new gate event row when person is seen on exit camera."""
    ph = "?" if _backend == "sqlite" else "%s"
    event_date = exit_dt.astimezone().strftime("%Y-%m-%d")
    exit_str = exit_dt.isoformat() if _backend == "sqlite" else exit_dt
    with _cursor(commit=True) as cur:
        cur.execute(
            f"INSERT INTO gate_events (person_name, event_date, exit_time) VALUES ({ph},{ph},{ph})",
            (person_name, event_date, exit_str),
        )
        if _backend == "sqlite":
            return cur.lastrowid
        cur.execute("SELECT lastval()")
        return cur.fetchone()[0]


def close_gate_entry(person_name, entry_dt):
    """Close the most recent open gate event for person when seen on entry camera."""
    ph = "?" if _backend == "sqlite" else "%s"
    entry_str = entry_dt.isoformat() if _backend == "sqlite" else entry_dt
    with _cursor(commit=True) as cur:
        # Find the newest open row (entry_time IS NULL)
        cur.execute(
            f"SELECT id, exit_time FROM gate_events WHERE person_name = {ph} AND entry_time IS NULL ORDER BY exit_time DESC LIMIT 1",
            (person_name,),
        )
        row = cur.fetchone()
        if row is None:
            return None
        row_id = row[0]
        exit_str_raw = row[1]
        try:
            if isinstance(exit_str_raw, str):
                from datetime import datetime as _dt, timezone as _tz
                exit_dt_stored = _dt.fromisoformat(exit_str_raw)
                if exit_dt_stored.tzinfo is None:
                    exit_dt_stored = exit_dt_stored.replace(tzinfo=_tz.utc)
            else:
                exit_dt_stored = exit_str_raw
            duration = round((entry_dt - exit_dt_stored).total_seconds() / 60, 1)
        except Exception:
            duration = None
        cur.execute(
            f"UPDATE gate_events SET entry_time={ph}, duration_minutes={ph} WHERE id={ph}",
            (entry_str, duration, row_id),
        )
        return row_id


def get_gate_events(date_str, person_name=None):
    """Return gate_events rows for a date, optionally filtered by person."""
    ph = "?" if _backend == "sqlite" else "%s"
    if person_name:
        sql = f"SELECT * FROM gate_events WHERE event_date = {ph} AND person_name = {ph} ORDER BY exit_time"
        params = (date_str, person_name)
    else:
        sql = f"SELECT * FROM gate_events WHERE event_date = {ph} ORDER BY person_name, exit_time"
        params = (date_str,)
    with _cursor() as cur:
        cur.execute(sql, params)
        return _rows_to_dicts(cur.fetchall())


def get_gate_events_range(date_from_str, date_to_str):
    """Return gate_events rows between two dates (inclusive)."""
    ph = "?" if _backend == "sqlite" else "%s"
    with _cursor() as cur:
        cur.execute(
            f"SELECT * FROM gate_events WHERE event_date >= {ph} AND event_date <= {ph} ORDER BY person_name, event_date, exit_time",
            (date_from_str, date_to_str),
        )
        return _rows_to_dicts(cur.fetchall())


# ---------------------------------------------------------------------------
# Zones
# ---------------------------------------------------------------------------

def get_zones(branch=None):
    ph = "?" if _backend == "sqlite" else "%s"
    with _cursor() as cur:
        if branch:
            cur.execute(f"SELECT * FROM zones WHERE branch = {ph} ORDER BY name", (branch,))
        else:
            cur.execute("SELECT * FROM zones ORDER BY name")
        return _rows_to_dicts(cur.fetchall())


def get_zone(zone_id):
    ph = "?" if _backend == "sqlite" else "%s"
    with _cursor() as cur:
        cur.execute(f"SELECT * FROM zones WHERE id = {ph}", (zone_id,))
        rows = _rows_to_dicts(cur.fetchall())
        return rows[0] if rows else None


def create_zone(name, description='', branch='Riyadh'):
    ph = "?" if _backend == "sqlite" else "%s"
    with _cursor(commit=True) as cur:
        cur.execute(
            f"INSERT INTO zones (name, description, branch) VALUES ({ph},{ph},{ph})",
            (name, description, branch),
        )
        if _backend == "sqlite":
            cur.execute("SELECT last_insert_rowid()")
        else:
            cur.execute("SELECT lastval()")
        return cur.fetchone()[0]


def update_zone(zone_id, name=None, description=None, branch=None):
    ph = "?" if _backend == "sqlite" else "%s"
    fields, params = [], []
    if name is not None:
        fields.append(f"name = {ph}"); params.append(name)
    if description is not None:
        fields.append(f"description = {ph}"); params.append(description)
    if branch is not None:
        fields.append(f"branch = {ph}"); params.append(branch)
    if not fields:
        return
    params.append(zone_id)
    with _cursor(commit=True) as cur:
        cur.execute(f"UPDATE zones SET {', '.join(fields)} WHERE id = {ph}", params)


def delete_zone(zone_id):
    ph = "?" if _backend == "sqlite" else "%s"
    with _cursor(commit=True) as cur:
        cur.execute(f"DELETE FROM zones WHERE id = {ph}", (zone_id,))


def get_zone_cameras(zone_id):
    ph = "?" if _backend == "sqlite" else "%s"
    with _cursor() as cur:
        cur.execute(
            f"SELECT l.id, l.name, l.camera_source FROM zone_cameras zc JOIN locations l ON l.id = zc.location_id WHERE zc.zone_id = {ph} ORDER BY l.name",
            (zone_id,),
        )
        return _rows_to_dicts(cur.fetchall())


def set_zone_cameras(zone_id, location_ids):
    ph = "?" if _backend == "sqlite" else "%s"
    with _cursor(commit=True) as cur:
        cur.execute(f"DELETE FROM zone_cameras WHERE zone_id = {ph}", (zone_id,))
        for loc_id in location_ids:
            cur.execute(f"INSERT INTO zone_cameras (zone_id, location_id) VALUES ({ph},{ph})", (zone_id, loc_id))


def get_zone_members(zone_id):
    ph = "?" if _backend == "sqlite" else "%s"
    with _cursor() as cur:
        cur.execute(f"SELECT name FROM people WHERE home_zone_id = {ph} ORDER BY name", (zone_id,))
        return [r["name"] for r in _rows_to_dicts(cur.fetchall())]


def set_person_home_zone(person_name, zone_id_or_null):
    ph = "?" if _backend == "sqlite" else "%s"
    with _cursor(commit=True) as cur:
        cur.execute(
            f"UPDATE people SET home_zone_id = {ph} WHERE name = {ph}",
            (zone_id_or_null, person_name),
        )


def get_zone_status_snapshot(away_threshold_minutes=30):
    """Return zone status for every person with a home zone assigned."""
    from datetime import datetime, timezone

    if _backend == "sqlite":
        sql = """
        WITH open_visits AS (
            SELECT v.person_name, v.location_id, v.last_seen, l.name AS location_name
            FROM visits v JOIN locations l ON l.id = v.location_id
            WHERE NOT v.ended
        ),
        open_gate_exits AS (
            SELECT person_name FROM gate_events
            WHERE event_date = date('now','localtime') AND entry_time IS NULL
        ),
        people_with_zone AS (
            SELECT p.name, p.home_zone_id, z.name AS home_zone_name, z.branch
            FROM people p JOIN zones z ON z.id = p.home_zone_id
            WHERE p.home_zone_id IS NOT NULL
        )
        SELECT
            pw.name,
            pw.home_zone_id,
            pw.home_zone_name,
            pw.branch,
            ov.location_id AS current_location_id,
            ov.location_name AS current_location_name,
            zc.zone_id AS current_zone_id,
            ov.last_seen AS last_seen_utc,
            CASE
                WHEN oge.person_name IS NOT NULL THEN 'out_of_building'
                WHEN ov.person_name IS NULL      THEN 'away'
                WHEN zc.zone_id = pw.home_zone_id THEN 'in_zone'
                ELSE 'out_of_zone'
            END AS status
        FROM people_with_zone pw
        LEFT JOIN open_visits ov ON ov.person_name = pw.name
        LEFT JOIN open_gate_exits oge ON oge.person_name = pw.name
        LEFT JOIN zone_cameras zc ON zc.location_id = ov.location_id
        ORDER BY pw.home_zone_name, pw.name
        """
    else:
        sql = """
        WITH open_visits AS (
            SELECT v.person_name, v.location_id, v.last_seen, l.name AS location_name
            FROM visits v JOIN locations l ON l.id = v.location_id
            WHERE NOT v.ended
        ),
        open_gate_exits AS (
            SELECT person_name FROM gate_events
            WHERE event_date = CURRENT_DATE AND entry_time IS NULL
        ),
        people_with_zone AS (
            SELECT p.name, p.home_zone_id, z.name AS home_zone_name, z.branch
            FROM people p JOIN zones z ON z.id = p.home_zone_id
            WHERE p.home_zone_id IS NOT NULL
        )
        SELECT
            pw.name,
            pw.home_zone_id,
            pw.home_zone_name,
            pw.branch,
            ov.location_id AS current_location_id,
            ov.location_name AS current_location_name,
            zc.zone_id AS current_zone_id,
            ov.last_seen AS last_seen_utc,
            CASE
                WHEN oge.person_name IS NOT NULL THEN 'out_of_building'
                WHEN ov.person_name IS NULL      THEN 'away'
                WHEN zc.zone_id = pw.home_zone_id THEN 'in_zone'
                ELSE 'out_of_zone'
            END AS status
        FROM people_with_zone pw
        LEFT JOIN open_visits ov ON ov.person_name = pw.name
        LEFT JOIN open_gate_exits oge ON oge.person_name = pw.name
        LEFT JOIN zone_cameras zc ON zc.location_id = ov.location_id
        ORDER BY pw.home_zone_name, pw.name
        """

    with _cursor() as cur:
        cur.execute(sql)
        rows = _rows_to_dicts(cur.fetchall())

    # Post-process: recently-seen person whose visit just closed → treat as in_zone
    now_utc = datetime.now(timezone.utc)
    for row in rows:
        if row['status'] == 'away' and row.get('last_seen_utc'):
            try:
                ls = row['last_seen_utc']
                if isinstance(ls, str):
                    ls = datetime.fromisoformat(ls.replace('Z', '+00:00'))
                if ls.tzinfo is None:
                    ls = ls.replace(tzinfo=timezone.utc)
                elapsed = (now_utc - ls).total_seconds() / 60
                if elapsed < away_threshold_minutes:
                    row['status'] = 'in_zone'
            except Exception:
                pass
    return rows


def get_zone_compliance_report(date_from, date_to, branch=None):
    """Return per-person per-day zone compliance stats between date_from and date_to (YYYY-MM-DD)."""
    ph = "?" if _backend == "sqlite" else "%s"

    if _backend == "sqlite":
        duration_in = f"(julianday(v.last_seen) - julianday(v.first_seen)) * 1440"
        day_expr = "date(v.first_seen, 'localtime')"
    else:
        duration_in = "EXTRACT(EPOCH FROM (v.last_seen::timestamptz - v.first_seen::timestamptz)) / 60.0"
        day_expr = "DATE(v.first_seen AT TIME ZONE 'localtime')"

    branch_filter = f" AND z.branch = {ph}" if branch else ""
    params = [date_from, date_to + " 23:59:59"]
    if branch:
        params.append(branch)

    sql = f"""
        SELECT
            v.person_name,
            z.name AS home_zone_name,
            z.id AS home_zone_id,
            {day_expr} AS day,
            SUM(CASE WHEN zc.zone_id = p.home_zone_id
                     THEN {duration_in} ELSE 0 END) AS minutes_in_zone,
            SUM(CASE WHEN zc.zone_id IS NULL OR zc.zone_id != p.home_zone_id
                     THEN {duration_in} ELSE 0 END) AS minutes_out_of_zone,
            COUNT(CASE WHEN zc.zone_id IS NULL OR zc.zone_id != p.home_zone_id THEN 1 END) AS out_of_zone_events
        FROM visits v
        JOIN people p ON p.name = v.person_name
        JOIN zones z ON z.id = p.home_zone_id
        LEFT JOIN zone_cameras zc ON zc.location_id = v.location_id
        WHERE v.first_seen >= {ph}
          AND v.first_seen <= {ph}
          AND v.person_name NOT LIKE 'unknown_%'
          AND p.home_zone_id IS NOT NULL
          {branch_filter}
        GROUP BY v.person_name, z.name, z.id, {day_expr}
        ORDER BY z.name, v.person_name, day
    """

    with _cursor() as cur:
        cur.execute(sql, params)
        return _rows_to_dicts(cur.fetchall())
