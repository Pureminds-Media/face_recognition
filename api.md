# Face Recognition API

A reference for building an external UI against this server. The local
Flask UI uses the same routes — anything the in-tree UI does, your client
can do too.

---

## 1. Hosting & exposing the server

The server runs on `http://localhost:5001` by default (`python app.py`).

### 1.1 Install ngrok

```bash
# Linux (snap)
sudo snap install ngrok

# OR Linux (deb)
curl -s https://ngrok-agent.s3.amazonaws.com/ngrok.asc \
  | sudo tee /etc/apt/trusted.gpg.d/ngrok.asc >/dev/null \
  && echo "deb https://ngrok-agent.s3.amazonaws.com buster main" \
  | sudo tee /etc/apt/sources.list.d/ngrok.list \
  && sudo apt update && sudo apt install ngrok

# macOS
brew install ngrok/ngrok/ngrok
```

### 1.2 Authenticate ngrok

Sign up at <https://dashboard.ngrok.com/signup>, copy your authtoken from
<https://dashboard.ngrok.com/get-started/your-authtoken>, then:

```bash
ngrok config add-authtoken <your-token>
```

This writes `~/.config/ngrok/ngrok.yml` once — you don't have to repeat
this on the same machine.

### 1.3 Generate an API key

```bash
python -c 'import secrets; print(secrets.token_urlsafe(32))'
```

Store the result in your `.env` (or export it inline):

```bash
# .env
API_KEY=<paste-the-generated-key>
```

### 1.4 Run the server + ngrok

In one terminal:

```bash
python app.py
```

In a second terminal:

```bash
ngrok http 5001
```

ngrok prints a public URL like `https://1a2b-3c4d.ngrok-free.app`. The
remote UI uses that URL as its API base.

### 1.5 Optional: persistent domain

Free-plan URLs change every restart. To pin a stable hostname:

- **Paid plan**: reserve a domain at
  <https://dashboard.ngrok.com/cloud-edge/domains>, then run
  `ngrok http --domain=your-app.ngrok.app 5001`.
- **Free plan**: each restart, update the remote client's base URL with
  the new hostname.

### 1.6 Optional: ngrok config file

Instead of long flags, put your tunnel in `~/.config/ngrok/ngrok.yml`:

```yaml
version: "3"
agent:
  authtoken: <your-token>
tunnels:
  face-recognition:
    proto: http
    addr: 5001
    # domain: your-app.ngrok.app   # uncomment on paid plan
```

Then start it with:

```bash
ngrok start face-recognition
```

### 1.7 Hand-off to the remote UI

Send the remote-UI builder:

1. The public ngrok URL.
2. The `API_KEY` value.
3. A pointer to this file (`api.md`) for endpoint details.

That's everything they need to call the API.

---

## 2. Authentication

When `API_KEY` is set, every `/api/*` and stream route requires the same
key. HTML page routes (`/`, `/settings`, …) are open so the local UI still
loads in a browser.

Two ways to send it:

- **Header** (preferred for JSON requests): `X-API-Key: <key>`
- **Query string** (for streams, `<img>`, `<video>`): `?api_key=<key>`

Missing or wrong key → `401 {"ok": false, "error": "unauthorized"}`.

If `API_KEY` is unset (default), no auth is enforced — keep this for local
development only.

---

## 3. Conventions

- All `/api/*` responses are JSON. Success bodies almost always start with
  `{"ok": true, …}`; errors return `{"ok": false, "error": "<msg>"}`
  with an HTTP 4xx/5xx status.
- Bodies for `POST` endpoints are JSON unless noted (file upload uses
  `multipart/form-data`).
- Timestamps are ISO 8601 strings unless noted.
- Person names are case-sensitive and normalised — stick to letters,
  digits, underscore, dash.

---

## 4. Endpoints

### 4.1 Engine state

| Method | Path           | Description                                           |
| ------ | -------------- | ----------------------------------------------------- |
| GET    | `/api/status`  | `{running, fps, cam_index, viewer_mode, …}`          |
| POST   | `/api/start`   | Start the detection engine. Returns immediately; camera connections open in the background. Poll `GET /api/status` (`running: true`) to confirm. |
| POST   | `/api/stop`    | Stop the engine.                                      |
| GET    | `/api/tracks`  | Live track snapshot: list of `{name, bbox, activity}`. |

### 4.2 People (face folders)

| Method | Path                                        | Description                                                 |
| ------ | ------------------------------------------- | ----------------------------------------------------------- |
| GET    | `/api/people`                               | List of `{name, count, thumbnail_url, section, branch}`.    |
| POST   | `/api/upload_face`                          | Upload one face image. `multipart/form-data`: `file`, `name` (existing or new). |
| POST   | `/api/bulk_upload_faces`                    | Bulk upload face images. `multipart/form-data`: `bulk_mode` (`"single_person"` or `"name_from_file"`), `files[]`. In `single_person` mode also pass `existing_name`/`new_name`/`mode` (same as `upload_face`). |
| POST   | `/api/rename_person`                        | Body: `{old_name, new_name}`. Renames folder and DB visits. |
| DELETE | `/api/person/<name>`                        | Delete a person and all their images.                       |
| GET    | `/api/person/<name>/images`                 | List `{filename, url}` for that person.                     |
| DELETE | `/api/person/<name>/image/<filename>`       | Delete one image.                                           |
| POST   | `/api/person/<name>/image/<file>/transfer`  | Body: `{target}`. Move one image to another person.         |
| POST   | `/api/person/<name>/images/bulk_delete`     | Body: `{filenames: [...]}`. Bulk delete.                    |
| POST   | `/api/person/<name>/images/bulk_transfer`   | Body: `{target, filenames: [...]}`. Bulk move.              |
| POST   | `/api/people/merge`                         | Body: `{sources: [...], target}`. Merge folders + DB visits. |
| POST   | `/api/reload_faces`                         | Force a synchronous embedding rebuild.                      |
| GET    | `/api/person/<name>/meta`                   | Returns `{name, section, branch, arabic_name, shift, home_zone_id}` for a person. |
| POST   | `/api/person/<name>/meta`                   | Body: `{section?, branch?, arabic_name?, shift?, home_zone_id?}`. Update fields. Auto-creates the row if missing. |

### 4.2.1 Sections

Sections are named groups (e.g. "IT", "HR") that people are assigned to. Managed from **Settings → Sections**.

| Method | Path                                        | Description                                                 |
| ------ | ------------------------------------------- | ----------------------------------------------------------- |
| GET    | `/api/sections`                             | List all sections. Returns `{sections: [{name, members: [...names]}]}`. |
| POST   | `/api/sections`                             | Body: `{name}`. Create a new section. Returns `{ok: true}`. |
| DELETE | `/api/sections/<name>`                      | Delete a section. Does NOT delete the assigned people; it clears their `section` field. |
| POST   | `/api/sections/<name>/rename`               | Body: `{new_name}`. Rename a section and update all members' `people.section` values atomically. Returns `{ok: true, new_name}`. |
| POST   | `/api/sections/<name>/assign`               | Body: `{person}`. Assign a person to this section (replaces any existing section). Returns `{ok: true}`. |
| POST   | `/api/sections/<name>/unassign`             | Body: `{person}`. Remove a person from this section (clears their section field). Returns `{ok: true}`. |
| POST   | `/api/sections/<name>/manager`              | Body: `{person}`. Set (or clear, if `person` is empty) the manager for this section. Returns `{ok: true}`. |

### 4.2.2 Zones

Zones are named camera zones (e.g. "Floor 1", "Server Room") with an optional branch. People can be assigned a *home zone*; the zone status endpoint shows who is currently inside vs. away from their zone.

| Method | Path                                        | Description                                                 |
| ------ | ------------------------------------------- | ----------------------------------------------------------- |
| GET    | `/api/zones`                                | List zones. Accepts `?branch=<name>` filter. Returns `{zones: [{id, name, description, branch, created_at}]}`. |
| POST   | `/api/zones`                                | Body: `{name, description?, branch?}`. Create a zone. Returns `{ok: true, id}`. |
| PUT    | `/api/zones/<zone_id>`                      | Body: `{name?, description?, branch?}`. Update a zone. |
| DELETE | `/api/zones/<zone_id>`                      | Delete a zone. Clears `people.home_zone_id` for all members. |
| GET    | `/api/zones/<zone_id>/cameras`              | List `{location_id, camera_source, name}` for cameras assigned to this zone. |
| POST   | `/api/zones/<zone_id>/cameras`              | Body: `{location_ids: [...]}`. Replace the full camera assignment for this zone. |
| GET    | `/api/zones/<zone_id>/members`              | List people with `home_zone_id` set to this zone. Returns `{members: [{name, …}]}`. |
| POST   | `/api/zones/<zone_id>/assign`               | Body: `{person_name}`. Set a person's home zone. Returns `{ok: true}`. |
| POST   | `/api/zones/<zone_id>/unassign`             | Body: `{person_name}`. Clear a person's home zone. Returns `{ok: true}`. |
| GET    | `/api/zones/status`                         | Live zone presence snapshot. Returns `{status: [{zone_id, zone_name, branch, members: [{name, status: "present"\|"away", last_seen, …}]}]}`. Status is derived from the `zone_away_threshold_minutes` setting in `reports_config.json`. Accepts `?branch=<name>` filter. |
| GET    | `/api/zones/report`                         | Zone compliance report. Requires `?date_from=YYYY-MM-DD&date_to=YYYY-MM-DD`. Accepts `?branch=<name>`. Returns `{rows: [{person_name, zone_name, days_present, days_absent, …}]}`. |

### 4.3 Cameras

| Method | Path                                          | Description                                                 |
| ------ | --------------------------------------------- | ----------------------------------------------------------- |
| GET    | `/api/camera`                                 | List devices + current viewer state. The `devices` list still includes `grid_RxC` layout entries; the in-tree UI now hides them since it operates in single-camera viewer mode only, but the layouts work via direct API calls. |
| POST   | `/api/camera`                                 | Body: `{source}`. Switch viewer to a camera URL/index, or to a `grid_RxC` layout (e.g. `"grid_2x2"`). Also accepts `{grid_offset: int}` to page through cameras when in grid mode. **Note:** the analysis pool always covers every configured camera regardless of viewer mode — switching viewer mode never starts/stops detection on any camera. |
| POST   | `/api/camera/reload`                          | Re-probe devices.                                           |
| GET    | `/api/camera/statuses`                        | Returns live/dead status for all cameras in the active grid. Map of `{source: {ok, last_frame_age_secs, …}}`. |
| POST   | `/api/camera/reconnect`                       | Body: `{source}`. Bypass the 2-minute reconnect backoff and immediately retry the given camera. |
| GET    | `/api/ip_cameras`                             | Configured IP-camera groups + cameras.                      |
| POST   | `/api/ip_cameras/groups`                      | Body: `{name, base_url?}`. Create group.                    |
| PUT    | `/api/ip_cameras/groups/<group_id>`           | Body: `{name?, base_url?}`. Update group.                   |
| DELETE | `/api/ip_cameras/groups/<group_id>`           | Delete group + cameras.                                     |
| POST   | `/api/ip_cameras/groups/<group_id>/cameras`   | Body: `{name, channel?}` or `{name, url}`. Add camera.      |
| PUT    | `/api/ip_cameras/cameras/<camera_id>`         | Body: `{name?, channel?, url?}`.                            |
| DELETE | `/api/ip_cameras/cameras/<camera_id>`         | Delete one IP camera.                                       |
| POST   | `/api/ip_cameras/cameras/<camera_id>/test`    | Probe RTSP and return resolution / error.                   |
| POST   | `/api/ip_cameras/groups/<group_id>/reorder`   | Body: `{order: ["cam_id", ...]}`. Reorder cameras within a group. Cameras not in the list keep their relative order after those that are. |
| GET    | `/api/grid/config`                            | Saved grid layout + slot assignments.                       |
| POST   | `/api/grid/config`                            | Body: `{layout: [rows, cols], slots: {…}}`. Save and apply. |

### 4.4 Visit history

All history endpoints accept an optional `?branch=<name>` query parameter (e.g. `?branch=Riyadh` or `?branch=Egypt`). When omitted, all branches are returned. The in-tree UI always appends the currently selected branch.

| Method | Path                                       | Description                                                 |
| ------ | ------------------------------------------ | ----------------------------------------------------------- |
| GET    | `/api/history/daily?date=YYYY-MM-DD`       | Visits on a single day.                                     |
| GET    | `/api/history/person/<name>?from=&to=`     | All visits for a person (date range optional).              |
| GET    | `/api/history/location/<id>?from=&to=`     | All visits at a location.                                   |
| GET    | `/api/history/locations`                   | All locations (with `display_name`).                        |
| GET    | `/api/history/persons`                     | Distinct person names with at least one visit.              |
| GET    | `/api/history/sessions`                    | Server run sessions.                                        |
| POST   | `/api/history/clear`                       | Wipe visits, sessions, and footage files. Destructive.      |

Visit objects include: `id, person_name, location_name, location_display, camera_source, branch, first_seen, last_seen, duration_secs, duration_fmt, ended, confidence, footage_url, activity`.

`branch` is set automatically when a visit is opened, derived from which IP camera group (and its `branch` field) the `camera_source` belongs to. All historical rows default to `"Riyadh"`.

`duration_secs` = `last_seen − first_seen` (wall-clock duration of the visit). It is **not** the on-camera/visible time. A visit with `duration_secs = 0` typically means a single-frame detection that closed before any subsequent frame refreshed `last_seen`. `visible_duration` (real on-camera seconds tracked by the footage writer) is used by the analytics `/longest` endpoint but is not currently exposed in the visit serializer.

### 4.5 Analytics

All analytics endpoints accept an optional `?branch=<name>` query parameter. When omitted, all branches are included. The in-tree UI always appends the active branch.

| Method | Path | Description |
| ------ | ---- | ----------- |
| GET | `/api/analytics/present_absent?date=YYYY-MM-DD` | Returns `{present: [...names], absent: [...names]}` for a given day. `present` = known persons with at least one visit on that day (filtered by `?branch=` if provided). `absent` = persons assigned to this branch in the `people` table with no visit that day. Without a branch filter, `absent` falls back to all enrolled known folders (`faces/`) with no visit that day. Used by the Present/Absent tile modals. |
| GET | `/api/analytics/summary?date=YYYY-MM-DD` | Single-request summary tiles for a given day (default today). Returns `{peak_hour, present_today, absent_today, unknowns_today}`. `peak_hour` is the local-time hour bucket with the most distinct people spotted (e.g. `"09:00 – 10:00"`), or `null` if no data. `present_today` is the count of distinct known persons with at least one visit today (branch-filtered). `absent_today` is the count of persons who have visited this branch historically but had no visit today; without a branch filter, counts all enrolled known folders minus present. `unknowns_today` is the count of `unknown_N` folders in `faces/` — unresolved auto-captured persons regardless of when they were last seen. |
| GET | `/api/analytics/earliest?date=YYYY-MM-DD` | Top 10 employees with the earliest first arrival on a given day (default today). Add `&order=latest` to get the 10 latest arrivals instead. Add `&shift=morning` (`work_start - 1h` → `work_end` local) or `&shift=night` (`night_work_start - 1h` → `night_work_end` next day local) to restrict to a shift window. Shift boundaries are configurable from **Settings → Advanced**. Night-shift results automatically exclude anyone who already appeared in the morning window (each person in at most one shift). Returns `{person_name, arrival_time}` rows. Excludes `unknown_N` names. The in-tree UI fetches both earliest and latest in parallel on load and caches them; the Earliest/Latest toggle switches between views without a new request. |
| GET | `/api/analytics/longest?period=day\|week\|month\|year` | Top 10 employees with the longest total on-camera duration for the period (calendar-aligned: week = Sun–Sat, month = 1st–last, year = Jan–Dec). Uses `visible_duration` when recorded, falls back to `last_seen − first_seen`. Returns `{person_name, total_secs, duration_fmt}` sorted descending. The in-tree UI renders this as an interactive horizontal bar chart (Chart.js). |
| GET | `/api/analytics/headcount?from=YYYY-MM-DD&to=YYYY-MM-DD` | Distinct people present per day over a date range (default: current month). Returns `{rows: [{date, count}]}` ordered by date ascending. Excludes `unknown_N`. |
| GET | `/api/analytics/heatmap?from=YYYY-MM-DD&to=YYYY-MM-DD` | Presence heatmap over a date range (default: current month). Returns `{dates, persons, present: {person: {date: true}}}`. The in-tree UI renders this as a scrollable employee × day grid with green cells for present days. |

### 4.5.1 Engine config

| Method | Path | Description |
| ------ | ---- | ----------- |
| GET | `/api/engine/config` | Returns current engine tuning: `{detect_every, motion_gate, motion_thresh, viewer_jpeg_quality, out_fps, high_priority_sources}`. |
| POST | `/api/engine/config` | Body: any subset of the above keys. Updates live without restart. `high_priority_sources` is an array of RTSP URL strings. |

### 4.5.2 Settings — Advanced (Shift Times)

| Method | Path | Description |
| ------ | ---- | ----------- |
| GET | `/api/advanced/config` | Returns shift time configuration from `reports_config.json`. Response: `{config: {work_start, work_end, late_threshold_minutes, night_shift_enabled, night_work_start, night_work_end, night_late_threshold_minutes}}`. |
| POST | `/api/advanced/config` | Body: any subset of the shift keys above. Saves to `reports_config.json`. |

### 4.5.3 Reports

| Method | Path | Description |
| ------ | ---- | ----------- |
| GET | `/api/reports/generate?date=YYYY-MM-DD` | Generate the gate report for a date. Returns `{rows: [{name, arrival, exits, status}]}`. `exits` is a list of `{exit_time, entry_time, duration_minutes}`. |
| GET | `/api/reports/history` | List saved daily reports. Returns `{dates: ["YYYY-MM-DD", …]}`. |
| GET | `/api/reports/history/<date>` | Retrieve a saved report for a specific date. |
| POST | `/api/reports/history/<date>/save` | Manually save the report for a date to the `daily_reports` table. |
| GET | `/api/reports/history/<date>/export` | Download the report as a CSV file. |

### 4.6 Attendance

| Method | Path                          | Description                                          |
| ------ | ----------------------------- | ---------------------------------------------------- |
| GET    | `/api/attendance`             | Roster snapshot: `{name, attended, present, …}`.     |
| POST   | `/api/attendance/reset`       | Clear in-memory attendance state.                    |
| GET    | `/api/attendance/stream`      | Server-Sent Events: `state`, `new`, `repeat`.        |

### 4.7 Test runner (offline video)

| Method | Path                              | Description                                        |
| ------ | --------------------------------- | -------------------------------------------------- |
| POST   | `/api/test/upload`                | `multipart/form-data`: `file`. Returns `job_id`.   |
| GET    | `/api/test/status/<job_id>`       | `{status, progress, result_url?, error?}`.         |
| GET    | `/test/results/<filename>`        | Download a finished output (key required).         |

### 4.8 Static assets (key required)

| Path                       | What                                              |
| -------------------------- | ------------------------------------------------- |
| `/video`                   | MJPEG stream of the live viewer feed.             |
| `/footage/<filename>`      | A saved visit footage clip.                       |
| `/faces/<person>/<file>`   | A face image from a person folder.                |

These are served as binary content. Embed them in `<img>`/`<video>` with
`?api_key=<key>` appended (browsers can't attach headers to those tags).

---

## 5. Streams

### Server-Sent Events (`/api/attendance/stream`)

Append `?api_key=<key>`. Standard `EventSource` works. Events:

- `state` — full roster snapshot on connect and after resets.
- `new` — `{name}`, fired when a person is marked attended.
- `repeat` — `{name}`, fired when an already-attended person is seen again.

### MJPEG (`/video`)

Append `?api_key=<key>`. Drop the URL into an `<img>` tag. The viewer
follows whatever camera/grid is currently active — switch via
`POST /api/camera`.

By default the MJPEG stream is **annotated** — bounding boxes and name
labels are drawn on the live feed. To turn this off (e.g. for crowded
scenes where overlapping boxes become illegible), set the server-side
env var `LIVE_ANNOTATIONS_ENABLED=0` and restart the server. Saved
footage clips remain annotated regardless of this flag. Clients that want to render their own overlay (e.g. a
selective highlight only on tapped people) can poll `GET /api/tracks`
or read the bbox field on `/api/attendance/stream`'s state events and
draw a transparent layer over the `<img>`.

In single-camera viewer mode the source-frame is downscaled to the
engine's `width × height` (defaults `1280 × 720`) before JPEG encoding,
so the bitrate stays reasonable even when the camera itself is 4K.
Recordings and face crops still use the camera's native resolution.

---

## 6. Examples

### List people

```bash
curl -H "X-API-Key: $API_KEY" https://<ngrok>/api/people
```

```javascript
const res = await fetch(`${BASE}/api/people`, {
  headers: { "X-API-Key": API_KEY },
});
const { people } = await res.json();
```

### Upload a face image

```javascript
const fd = new FormData();
fd.append("file", file);
fd.append("name", "alice");
await fetch(`${BASE}/api/upload_face`, {
  method: "POST",
  headers: { "X-API-Key": API_KEY },  // do NOT set Content-Type with FormData
  body: fd,
});
```

### Today's visit history

```javascript
const date = new Date().toISOString().slice(0, 10);
const res = await fetch(`${BASE}/api/history/daily?date=${date}`, {
  headers: { "X-API-Key": API_KEY },
});
const { visits } = await res.json();
```

### Merge folders

```javascript
await fetch(`${BASE}/api/people/merge`, {
  method: "POST",
  headers: { "X-API-Key": API_KEY, "Content-Type": "application/json" },
  body: JSON.stringify({ sources: ["unknown_3", "unknown_7"], target: "alice" }),
});
```

### Embed the live feed

```html
<img src="https://<ngrok>/video?api_key=<key>" />
```

### Listen to attendance events

```javascript
const es = new EventSource(`${BASE}/api/attendance/stream?api_key=${API_KEY}`);
es.addEventListener("state", (e) => console.log("roster", JSON.parse(e.data)));
es.addEventListener("new",   (e) => console.log("attended", JSON.parse(e.data)));
```

---

## 7. Security notes

- The API key is shared-secret. Anyone holding it can mutate state —
  rotate it (change the env var, restart) if you suspect leakage.
- Destructive endpoints (`/api/history/clear`, `/api/person/<name>` DELETE,
  `/api/people/merge`) are gated only by the API key. Build confirmation
  prompts into your UI.
- ngrok free-plan URLs rotate; the remote client needs to update its base
  URL each restart. Use a paid ngrok subdomain to keep a stable hostname.
- MJPEG over a public tunnel is bandwidth-heavy. If the live feed isn't
  needed remotely, run ngrok only when needed and keep `/video` to the
  local network.
