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
    activity        TEXT
);

CREATE INDEX IF NOT EXISTS idx_visits_person     ON visits (person_name);
CREATE INDEX IF NOT EXISTS idx_visits_location   ON visits (location_id);
CREATE INDEX IF NOT EXISTS idx_visits_first_seen ON visits (first_seen);
CREATE INDEX IF NOT EXISTS idx_visits_open       ON visits (person_name, location_id) WHERE NOT ended;
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
    activity        TEXT
);

CREATE INDEX IF NOT EXISTS idx_visits_person     ON visits (person_name);
CREATE INDEX IF NOT EXISTS idx_visits_location   ON visits (location_id);
CREATE INDEX IF NOT EXISTS idx_visits_first_seen ON visits (first_seen);
CREATE INDEX IF NOT EXISTS idx_visits_open       ON visits (person_name, location_id) WHERE NOT ended;
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

def open_visit(person_name, location_id, confidence=None, session_id=None):
    """Create a new open visit. Returns the visit id."""
    now = _now_str() if _backend == "sqlite" else _now()
    ended_val = 0 if _backend == "sqlite" else False
    with _cursor(commit=True) as cur:
        if _backend == "postgres":
            cur.execute(
                """
                INSERT INTO visits (person_name, location_id, first_seen, last_seen,
                                    ended, confidence, session_id)
                VALUES (%s, %s, %s, %s, FALSE, %s, %s)
                RETURNING id
                """,
                (person_name, location_id, now, now, confidence, session_id),
            )
            return cur.fetchone()["id"]
        else:
            cur.execute(
                """
                INSERT INTO visits (person_name, location_id, first_seen, last_seen,
                                    ended, confidence, session_id)
                VALUES (?, ?, ?, ?, 0, ?, ?)
                """,
                (person_name, location_id, now, now, confidence, session_id),
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

def get_person_visits(person_name, date_from=None, date_to=None, limit=200):
    """All visits for a person, optionally filtered by date range."""
    ph = "?" if _backend == "sqlite" else "%s"
    clauses = [f"v.person_name = {ph}"]
    params = [person_name]
    if date_from:
        clauses.append(f"v.first_seen >= {ph}")
        params.append(date_from if _backend == "postgres" else str(date_from))
    if date_to:
        clauses.append(f"v.first_seen < {ph}")
        params.append(date_to if _backend == "postgres" else str(date_to))
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


def get_location_visits(location_id, date_from=None, date_to=None, limit=200):
    """All visits at a location, optionally filtered by date range."""
    ph = "?" if _backend == "sqlite" else "%s"
    clauses = [f"v.location_id = {ph}"]
    params = [location_id]
    if date_from:
        clauses.append(f"v.first_seen >= {ph}")
        params.append(date_from if _backend == "postgres" else str(date_from))
    if date_to:
        clauses.append(f"v.first_seen < {ph}")
        params.append(date_to if _backend == "postgres" else str(date_to))
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


def get_daily_summary(date):
    """All visits for a calendar day, grouped by person then time.

    Parameters
    ----------
    date : datetime.date
        The calendar day to query.

    Returns a list of dicts sorted by first_seen descending (most recent first).
    """
    day_start = datetime(date.year, date.month, date.day, tzinfo=timezone.utc)
    day_end = day_start + timedelta(days=1)

    if _backend == "postgres":
        params = (day_start, day_end)
    else:
        params = (day_start.isoformat(), day_end.isoformat())

    ph = "?" if _backend == "sqlite" else "%s"
    sql = f"""
        SELECT v.id, v.person_name, v.first_seen, v.last_seen, v.ended,
               v.confidence, v.screenshot, v.footage, v.visible_duration, v.activity,
               l.name as location_name, l.camera_source
        FROM visits v
        JOIN locations l ON l.id = v.location_id
        WHERE v.first_seen >= {ph} AND v.first_seen < {ph}
        ORDER BY v.first_seen DESC
    """
    with _cursor() as cur:
        cur.execute(sql, params)
        return _rows_to_dicts(cur.fetchall())


def get_known_persons(limit=500):
    """Return distinct person names that have visits."""
    ph = "?" if _backend == "sqlite" else "%s"
    sql = f"SELECT DISTINCT person_name FROM visits ORDER BY person_name LIMIT {ph}"
    with _cursor() as cur:
        cur.execute(sql, (limit,))
        return [dict(r)["person_name"] for r in cur.fetchall()]


def clear_all_data():
    """Delete all visits and sessions. Locations are kept (they map to cameras).

    Returns the number of visits deleted.
    """
    with _cursor(commit=True) as cur:
        cur.execute("DELETE FROM visits")
        visit_count = cur.rowcount
        cur.execute("DELETE FROM sessions")
    return visit_count
