# Face Recognition API

A reference for building an external UI against this server. The local
Flask UI uses the same routes — anything the in-tree UI does, your client
can do too.

---

## 1. Hosting & exposing the server

The server runs on `http://localhost:5000` by default (`python app.py`).

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
ngrok http 5000
```

ngrok prints a public URL like `https://1a2b-3c4d.ngrok-free.app`. The
remote UI uses that URL as its API base.

### 1.5 Optional: persistent domain

Free-plan URLs change every restart. To pin a stable hostname:

- **Paid plan**: reserve a domain at
  <https://dashboard.ngrok.com/cloud-edge/domains>, then run
  `ngrok http --domain=your-app.ngrok.app 5000`.
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
    addr: 5000
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
| POST   | `/api/start`   | Start the detection engine.                           |
| POST   | `/api/stop`    | Stop the engine.                                      |
| GET    | `/api/tracks`  | Live track snapshot: list of `{name, bbox, activity}`. |

### 4.2 People (face folders)

| Method | Path                                        | Description                                                 |
| ------ | ------------------------------------------- | ----------------------------------------------------------- |
| GET    | `/api/people`                               | List of `{name, count, thumbnail_url}`.                     |
| POST   | `/api/upload_face`                          | Upload one face image. `multipart/form-data`: `file`, `name` (existing or new). |
| POST   | `/api/rename_person`                        | Body: `{old_name, new_name}`. Renames folder and DB visits. |
| DELETE | `/api/person/<name>`                        | Delete a person and all their images.                       |
| GET    | `/api/person/<name>/images`                 | List `{filename, url}` for that person.                     |
| DELETE | `/api/person/<name>/image/<filename>`       | Delete one image.                                           |
| POST   | `/api/person/<name>/image/<file>/transfer`  | Body: `{target}`. Move one image to another person.         |
| POST   | `/api/person/<name>/images/bulk_delete`     | Body: `{filenames: [...]}`. Bulk delete.                    |
| POST   | `/api/person/<name>/images/bulk_transfer`   | Body: `{target, filenames: [...]}`. Bulk move.              |
| POST   | `/api/people/merge`                         | Body: `{sources: [...], target}`. Merge folders + DB visits. |
| POST   | `/api/reload_faces`                         | Force a synchronous embedding rebuild.                      |

### 4.3 Cameras

| Method | Path                                          | Description                                                 |
| ------ | --------------------------------------------- | ----------------------------------------------------------- |
| GET    | `/api/camera`                                 | List devices + current viewer state. The `devices` list still includes `grid_RxC` layout entries; the in-tree UI now hides them since it operates in single-camera viewer mode only, but the layouts work via direct API calls. |
| POST   | `/api/camera`                                 | Body: `{source}`. Switch viewer to a camera URL/index, or to a `grid_RxC` layout (e.g. `"grid_2x2"`). Also accepts `{grid_offset: int}` to page through cameras when in grid mode. **Note:** the analysis pool always covers every configured camera regardless of viewer mode — switching viewer mode never starts/stops detection on any camera. |
| POST   | `/api/camera/reload`                          | Re-probe devices.                                           |
| GET    | `/api/ip_cameras`                             | Configured IP-camera groups + cameras.                      |
| POST   | `/api/ip_cameras/groups`                      | Body: `{name, base_url?}`. Create group.                    |
| PUT    | `/api/ip_cameras/groups/<group_id>`           | Body: `{name?, base_url?}`. Update group.                   |
| DELETE | `/api/ip_cameras/groups/<group_id>`           | Delete group + cameras.                                     |
| POST   | `/api/ip_cameras/groups/<group_id>/cameras`   | Body: `{name, channel?}` or `{name, url}`. Add camera.      |
| PUT    | `/api/ip_cameras/cameras/<camera_id>`         | Body: `{name?, channel?, url?}`.                            |
| DELETE | `/api/ip_cameras/cameras/<camera_id>`         | Delete one IP camera.                                       |
| POST   | `/api/ip_cameras/cameras/<camera_id>/test`    | Probe RTSP and return resolution / error.                   |
| GET    | `/api/grid/config`                            | Saved grid layout + slot assignments.                       |
| POST   | `/api/grid/config`                            | Body: `{layout: [rows, cols], slots: {…}}`. Save and apply. |

### 4.4 Visit history

| Method | Path                                       | Description                                                 |
| ------ | ------------------------------------------ | ----------------------------------------------------------- |
| GET    | `/api/history/daily?date=YYYY-MM-DD`       | Visits on a single day.                                     |
| GET    | `/api/history/person/<name>?from=&to=`     | All visits for a person (date range optional).              |
| GET    | `/api/history/location/<id>?from=&to=`     | All visits at a location.                                   |
| GET    | `/api/history/locations`                   | All locations (with `display_name`).                        |
| GET    | `/api/history/persons`                     | Distinct person names with at least one visit.              |
| GET    | `/api/history/sessions`                    | Server run sessions.                                        |
| POST   | `/api/history/clear`                       | Wipe visits, sessions, screenshots, footage. Destructive.   |

Visit objects include: `id, person_name, location_name, location_display, camera_source, first_seen, last_seen, duration_secs, duration_fmt, ended, confidence, screenshot_url, footage_url, activity`.

`duration_secs` = `last_seen − first_seen` (wall-clock duration of the visit). It is **not** the on-camera/visible time. A visit with `duration_secs = 0` typically means a single-frame detection that closed before any subsequent frame refreshed `last_seen`. The `visible_duration` column tracked by the footage writer (real on-camera seconds) is currently not exposed in the visit serializer.

### 4.5 Attendance

| Method | Path                          | Description                                          |
| ------ | ----------------------------- | ---------------------------------------------------- |
| GET    | `/api/attendance`             | Roster snapshot: `{name, attended, present, …}`.     |
| POST   | `/api/attendance/reset`       | Clear in-memory attendance state.                    |
| GET    | `/api/attendance/stream`      | Server-Sent Events: `state`, `new`, `repeat`.        |

### 4.6 Test runner (offline video)

| Method | Path                              | Description                                        |
| ------ | --------------------------------- | -------------------------------------------------- |
| POST   | `/api/test/upload`                | `multipart/form-data`: `file`. Returns `job_id`.   |
| GET    | `/api/test/status/<job_id>`       | `{status, progress, result_url?, error?}`.         |
| GET    | `/test/results/<filename>`        | Download a finished output (key required).         |

### 4.7 Static assets (key required)

| Path                       | What                                              |
| -------------------------- | ------------------------------------------------- |
| `/video`                   | MJPEG stream of the live viewer feed.             |
| `/screenshots/<filename>`  | A saved visit screenshot.                         |
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

The MJPEG stream is intentionally **clean** — no per-person bounding
boxes or name labels are drawn on the live feed (they cluttered the
view when many people were present). Boxes/labels remain on the saved
footage clips and per-visit screenshots for after-the-fact review. If
your client UI wants to overlay boxes, poll `GET /api/tracks` (or read
the bbox field on `/api/attendance/stream`'s state events) and render
them as a transparent layer over the `<img>`.

In single-camera viewer mode the source-frame is downscaled to the
engine's `width × height` (defaults `1280 × 720`) before JPEG encoding,
so the bitrate stays reasonable even when the camera itself is 4K.
Recordings, screenshots and face crops still use the camera's native
resolution.

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
