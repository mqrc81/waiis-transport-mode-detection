# Metadata

Hand-curated facts the WAIIS data do not record. In git (small, reviewed by diff; only pseudonymous user keys); the
data themselves stay outside git. Read by the shared reader, so it doesn't belong to any one experiment.

## `waiis_devices.csv`

The phone behind each WAIIS user key, until the app stores platform and phone model with the data. Read by
`tmd.readers.waiis.device(user_key, date)`, which fails when a recorded day has no row or more than one.

| Column | Meaning |
|---|---|
| `user_key` | the `user=` key of the raw data folders |
| `valid_from`, `valid_to` | UTC dates (inclusive) the row covers; `valid_to` empty = still current. A phone change is a new row. |
| `platform` | `iphone` or `android` (as the benchmark) |
| `manufacturer`, `phone_model` | as reported, e.g. `Apple`, `iPhone 16` |
| `axis_convention` | the sign convention of the app's accelerometer axes: `ios` (iOS signs, scaled to m/s²) or `android`; `unknown` makes the reader fail |
| `axis_evidence` | how the convention is known (e.g. a face-up recording, or inferred) |
| `source`, `confirmed_on` | who reported the device, and the date it was confirmed |
| `notes` | free text |
