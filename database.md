# Database Reference

The system supports two backends behind the same Python API (`db.py`). The schema is identical in structure; only type names and placeholder syntax differ.

| Backend | How to activate | Notes |
|---------|-----------------|-------|
| **SQLite** | Default — no config needed | File at `DATABASE_PATH` (default `face_recognition.db`). WAL mode, `wal_autocheckpoint=1000`. |
| **PostgreSQL** | Set `DATABASE_URL=postgresql://…` | Uses a `psycopg2` thread pool (1–5 connections). |

---

## Tables

### `locations`

Maps a camera source string to a human-readable name. One row per unique camera that has ever been active.

| Column | SQLite type | PG type | Notes |
|--------|------------|---------|-------|
| `id` | INTEGER PK AUTOINCREMENT | SERIAL PK | |
| `camera_source` | TEXT NOT NULL UNIQUE | TEXT NOT NULL UNIQUE | RTSP URL, webcam index, etc. |
| `name` | TEXT NOT NULL | TEXT NOT NULL | Display name (e.g. "Entrance") |
| `created_at` | TEXT DEFAULT datetime('now') | TIMESTAMPTZ DEFAULT NOW() | |

---

### `sessions`

One row per engine run (start → stop).

| Column | SQLite type | PG type | Notes |
|--------|------------|---------|-------|
| `id` | TEXT PK | UUID PK | UUID string |
| `started_at` | TEXT NOT NULL | TIMESTAMPTZ NOT NULL | Engine start time |
| `ended_at` | TEXT | TIMESTAMPTZ | NULL while running |
| `camera_source` | TEXT | TEXT | Primary camera at session start |

---

### `visits`

Core attendance table. One row per continuous presence of one person at one camera location.

| Column | SQLite type | PG type | Notes |
|--------|------------|---------|-------|
| `id` | INTEGER PK AUTOINCREMENT | BIGSERIAL PK | |
| `person_name` | TEXT NOT NULL | TEXT NOT NULL | Matches `faces/<name>/` directory |
| `location_id` | INTEGER → `locations.id` | INTEGER → `locations.id` | FK; NULL if location not yet registered |
| `first_seen` | TEXT NOT NULL | TIMESTAMPTZ NOT NULL | UTC timestamp — visit open time |
| `last_seen` | TEXT NOT NULL | TIMESTAMPTZ NOT NULL | UTC timestamp — last confirmed detection |
| `ended` | INTEGER DEFAULT 0 | BOOLEAN DEFAULT FALSE | 0/false = open, 1/true = closed |
| `confidence` | REAL | FLOAT | Lowest (best) ArcFace cosine distance seen |
| `session_id` | TEXT → `sessions.id` | UUID → `sessions.id` | FK |
| `screenshot` | TEXT | TEXT | Face crop filename, served at `/faces/` |
| `footage` | TEXT | TEXT | WebM clip filename, served at `/footage/` |
| `visible_duration` | REAL | FLOAT | Seconds actually tracked (footage clock) |
| `activity` | TEXT | TEXT | Most frequent CLIP action label |
| `branch` | TEXT NOT NULL DEFAULT 'Riyadh' | TEXT NOT NULL DEFAULT 'Riyadh' | Auto-assigned from the camera's IP group |

**Indexes:**

| Index | Columns | Condition |
|-------|---------|-----------|
| `idx_visits_person` | `person_name` | |
| `idx_visits_location` | `location_id` | |
| `idx_visits_first_seen` | `first_seen` | |
| `idx_visits_open` | `person_name, location_id` | `WHERE NOT ended` |

---

### `people`

Per-person metadata. Created on first meta save; a person can exist in `faces/` without a row here.

| Column | SQLite type | PG type | Notes |
|--------|------------|---------|-------|
| `name` | TEXT PK | TEXT PK | Matches `faces/<name>/` directory |
| `section` | TEXT NOT NULL DEFAULT '' | TEXT NOT NULL DEFAULT '' | Section name; '' = unassigned |
| `branch` | TEXT NOT NULL DEFAULT 'Riyadh' | TEXT NOT NULL DEFAULT 'Riyadh' | Home branch for absent-list filtering |
| `email` | TEXT NOT NULL DEFAULT '' | TEXT NOT NULL DEFAULT '' | Optional contact email |
| `arabic_name` | TEXT NOT NULL DEFAULT '' | TEXT NOT NULL DEFAULT '' | Arabic display name |
| `shift` | TEXT NOT NULL DEFAULT '' | TEXT NOT NULL DEFAULT '' | Shift assignment (e.g. `'morning'`, `'night'`); '' = not assigned |
| `home_zone_id` | INTEGER → `zones.id` | INTEGER → `zones.id` | FK (nullable) — the zone this person is expected to be in |

---

### `sections`

Named groups people can be assigned to (e.g. "IT", "HR", "Security").

| Column | SQLite type | PG type | Notes |
|--------|------------|---------|-------|
| `id` | INTEGER PK AUTOINCREMENT | SERIAL PK | |
| `name` | TEXT NOT NULL UNIQUE | TEXT NOT NULL UNIQUE | Section display name |
| `manager` | TEXT NOT NULL DEFAULT '' | TEXT NOT NULL DEFAULT '' | Name of the manager person (matches `people.name`); '' = no manager set |

**Relation:** `people.section` mirrors `sections.name`. When a section is renamed, both `sections.name` and all matching `people.section` values are updated atomically. When a section is deleted, `people.section` is cleared to `''`.

---

### `zones`

Named camera zones (e.g. "Floor 1", "Server Room"). People can have a `home_zone_id` pointing here; the zone status API uses this to determine presence vs. away.

| Column | SQLite type | PG type | Notes |
|--------|------------|---------|-------|
| `id` | INTEGER PK AUTOINCREMENT | SERIAL PK | |
| `name` | TEXT NOT NULL UNIQUE | TEXT NOT NULL UNIQUE | Zone display name |
| `description` | TEXT NOT NULL DEFAULT '' | TEXT NOT NULL DEFAULT '' | Optional description |
| `branch` | TEXT NOT NULL DEFAULT 'Riyadh' | TEXT NOT NULL DEFAULT 'Riyadh' | Branch this zone belongs to |
| `created_at` | TEXT DEFAULT datetime('now') | TIMESTAMPTZ DEFAULT NOW() | |

---

### `zone_cameras`

Many-to-many join between zones and locations (cameras). A zone can cover multiple cameras; a camera can belong to multiple zones.

| Column | SQLite type | PG type | Notes |
|--------|------------|---------|-------|
| `zone_id` | INTEGER → `zones.id` ON DELETE CASCADE | INTEGER → `zones.id` ON DELETE CASCADE | |
| `location_id` | INTEGER → `locations.id` ON DELETE CASCADE | INTEGER → `locations.id` ON DELETE CASCADE | |

**Primary key:** `(zone_id, location_id)`.

---

### `gate_events`

Records exit/entry pairs through the two designated gate cameras. Written live as people pass through.

| Column | SQLite type | PG type | Notes |
|--------|------------|---------|-------|
| `id` | INTEGER PK AUTOINCREMENT | SERIAL PK | |
| `person_name` | TEXT NOT NULL | TEXT NOT NULL | Matches `faces/<name>/` |
| `event_date` | TEXT NOT NULL | TEXT NOT NULL | `YYYY-MM-DD` — calendar date of the exit |
| `exit_time` | TEXT NOT NULL | TEXT NOT NULL | ISO timestamp (UTC) — when person hit exit camera |
| `entry_time` | TEXT | TEXT | ISO timestamp (UTC) — when person returned; NULL = still out |
| `duration_minutes` | REAL | FLOAT | `entry_time − exit_time` in minutes; NULL until entry recorded |

**Index:** `idx_gate_events_person_date` on `(person_name, event_date)`.

**Write rules:**
- A row is **opened** (only `exit_time` set) each time a person is detected on the exit camera.
- The most-recent open row for that person is **closed** (`entry_time` + `duration_minutes`) when they appear on the arrival camera.
- If they leave again before returning, a new row is opened — multiple rows per person per day are expected.

---

### `daily_reports`

Auto-saved nightly snapshot of the gate report (saved at 23:00 by the scheduler thread).

| Column | SQLite type | PG type | Notes |
|--------|------------|---------|-------|
| `id` | INTEGER PK AUTOINCREMENT | SERIAL PK | |
| `report_date` | TEXT NOT NULL UNIQUE | TEXT NOT NULL UNIQUE | `YYYY-MM-DD` |
| `generated_at` | TEXT NOT NULL DEFAULT datetime('now') | TEXT NOT NULL | ISO timestamp when saved |
| `arrival_camera` | TEXT NOT NULL DEFAULT '' | TEXT NOT NULL DEFAULT '' | RTSP URL of arrival camera at save time |
| `exit_camera` | TEXT NOT NULL DEFAULT '' | TEXT NOT NULL DEFAULT '' | RTSP URL of exit camera at save time |
| `work_start` | TEXT NOT NULL DEFAULT '08:00' | TEXT NOT NULL DEFAULT '08:00' | Work start time from config |
| `late_threshold` | INTEGER NOT NULL DEFAULT 15 | INTEGER NOT NULL DEFAULT 15 | Minutes after work_start = late |
| `total_people` | INTEGER NOT NULL DEFAULT 0 | INTEGER NOT NULL DEFAULT 0 | Total enrolled people in report |
| `late_arrivals` | INTEGER NOT NULL DEFAULT 0 | INTEGER NOT NULL DEFAULT 0 | Count of people who arrived late |
| `late_exits` | INTEGER NOT NULL DEFAULT 0 | INTEGER NOT NULL DEFAULT 0 | Count of people who exited late |
| `records_json` | TEXT NOT NULL DEFAULT '[]' | TEXT NOT NULL DEFAULT '[]' | Full serialised report rows (JSON array) |
| `sent_email` | INTEGER NOT NULL DEFAULT 0 | INTEGER NOT NULL DEFAULT 0 | 1 if the daily email was sent |

**Index:** `idx_daily_reports_date` on `report_date`.

---

## Entity Relationship

```
sessions ──────────────────────────────────┐
   id (UUID/TEXT)                          │
                                           │ session_id (FK)
locations ─────────────────────────────────┤
   id (int)          camera_source         │
        │                                  │
        │ location_id (FK)                 │
        ▼                                  ▼
      visits ─────────────────────────────────
        person_name ──────────────────────────┐
                                              │ (soft link — no FK)
people ───────────────────────────────────────┘
   name (PK)
   section ──────────── (mirrors) ──── sections.name
   branch
   arabic_name, shift
   home_zone_id ──────────────────── zones.id (FK nullable)

sections
   name (PK-unique)
   manager ──────────── (soft link) ── people.name

zones
   id (PK)
   name, description, branch
        │  zone_cameras (join table)
        └───────────────────────────── locations.id

gate_events
   person_name ─────── (soft link) ── people.name / visits.person_name

daily_reports
   (standalone snapshot, no FK relations)
```

**Soft links** (not enforced by FK constraints): `people.name` ↔ `visits.person_name`, `people.section` ↔ `sections.name`, `gate_events.person_name` ↔ `people.name`, `sections.manager` ↔ `people.name`. These are kept consistent in application code rather than by the database.

---

## Migrations

Additive migrations run at startup inside `init_db()` using `ALTER TABLE … ADD COLUMN`, wrapped in try/except so re-running on an already-migrated database is safe.

| Column added | Migration notes |
|---|---|
| `visits.branch` | Back-fills all existing rows to `'Riyadh'` |
| `visits.screenshot`, `footage`, `visible_duration`, `activity` | Added silently if missing |
| `people` table | Created if missing (pre-sections databases) |
| `sections` table | Created if missing |
| `sections.manager` | Added if missing |
| `daily_reports` table | Created if missing |
| `gate_events` table | Created if missing |
| `people.email` | Added if missing |
| `people.arabic_name` | Added if missing |
| `people.shift` | Added if missing |
| `people.home_zone_id` | Added if missing; FK to `zones.id` |
| `zones` table | Created if missing |
| `zone_cameras` table | Created if missing |

---

## File-based state (not in DB)

Some state is stored as files alongside the database:

| File | Contents |
|------|----------|
| `ip_cameras.json` | Camera groups, channel numbers, resolved RTSP URLs |
| `reports_config.json` | Arrival/exit camera URLs, work hours, email config |
| `grid_config.json` | Grid layout and slot assignments |
| `faces/<name>/*.jpg` | Enrolled face images |
| `faces/<name>/.arcface.npz` | Cached ArcFace embeddings (auto-generated) |
