# PostgreSQL setup

PostgreSQL 18 is installed and its Windows service is running. The server asks
for the `postgres` password, so database creation must be completed with the
credentials set during installation.

In pgAdmin's Query Tool, connected as an administrator, run:

```sql
CREATE ROLE tennis_app WITH LOGIN PASSWORD 'replace-with-a-strong-password';
CREATE DATABASE tennis_point_predictor OWNER tennis_app;
```

Then set the URL only in your shell (or a local ignored `.env` file):

```powershell
$env:DATABASE_URL = 'postgresql+psycopg://tennis_app:PASSWORD@localhost:5432/tennis_point_predictor'
```

The pipeline creates tables automatically. To create/review them manually,
connect to `tennis_point_predictor` and run `schema.sql`.

Useful checks after a run:

```sql
SELECT v.id, v.source_path, count(f.id) AS frames
FROM videos v LEFT JOIN frames f ON f.video_id = v.id
GROUP BY v.id ORDER BY v.id DESC;

SELECT f.frame_index, p.player_id, p.court_x, p.court_y, p.confidence
FROM player_detections p JOIN frames f ON f.id = p.frame_id
WHERE f.video_id = 1
ORDER BY f.frame_index, p.player_id;
```

Frame and player rows are inserted in batches on a background thread. A failed
writer is surfaced back to the main process on the next submit or shutdown.
