# Raw data — sources, licences and the two files you must fetch yourself

Every file the pipeline reads from `data/raw/` is listed below. Six are
redistributable and are committed here. Two are not, and have been removed; the
instructions for obtaining them are in the second table.

## Included

| File | Source | Licence | Attribution required |
|---|---|---|---|
| `elec_price_hourly_ember.csv` | Ember, *European Wholesale Electricity Price Data* | CC BY 4.0 | Yes, see below |
| `elec_demand_outturn_hh_bmrs.csv` | Elexon BMRS / NESO historic demand outturn | Elexon/NESO open terms &mdash; **not verified**, see below | Source acknowledgement |
| `elec_demand_forecast_da_hh_bmrs.csv` | Elexon BMRS day-ahead demand forecast | as above &mdash; **not verified** | Source acknowledgement |
| `renew_gen_forecast_da_hourly_bmrs.csv` | Elexon BMRS day-ahead wind and solar forecast | as above &mdash; **not verified** | Source acknowledgement |
| `renew_repd_sites_desnz.csv` | DESNZ, *Renewable Energy Planning Database* (REPD) | Open Government Licence v3.0 | Yes, see below |
| `weather_forecast_da_hourly_openmeteo.csv` | Open-Meteo Previous Model Runs API | CC BY 4.0 (non-commercial API use) | Source acknowledgement |

`wind_gen_predicted_hourly_xgboost.csv` and `solar_gen_predicted_hourly_xgboost.csv`
also sit in this directory but are not source data: they are the Stage-1 model
outputs, produced by `notebooks/02_weather_processing_gen_prediction.ipynb`.

### The three Elexon BMRS / NESO licences are not verified

Those files were downloaded for free re-use under what were understood to be
Elexon's and NESO's open data terms, but no licence name is recorded anywhere in
this tree and the current terms have not been re-checked against the publishers'
sites. Treat them as unresolved until you confirm them yourself. The other three
included files name a specific licence and are as stated.

### Ember attribution (CC BY 4.0)

> Electricity price data from Ember, *European Wholesale Electricity Price Data*
> (https://ember-energy.org/data/european-wholesale-electricity-price-data/),
> licensed under CC BY 4.0.

The file was originally committed under the name `elec_price_hourly_entsoe.csv`.
That name was wrong. The column schema — `Country, ISO3 Code, Datetime (UTC),
Datetime (Local), Price (EUR/MWhe)` — is Ember's, not an ENTSO-E Transparency
Platform export. Ember compiles the series from ENTSO-E and republishes it under
CC BY 4.0, which is what makes redistribution here permissible. The file has been
renamed to `elec_price_hourly_ember.csv` and every reference in the code and the
notebooks updated.

### DESNZ REPD attribution (Open Government Licence v3.0)

> Contains information from the Renewable Energy Planning Database (REPD),
> Department for Energy Security and Net Zero, licensed under the
> Open Government Licence v3.0
> (https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/).

## Removed — obtain these two yourself

Both are commercially licensed and cannot be redistributed. Stage 2 will raise
`FileNotFoundError` until they are placed in this directory under exactly these
names.

| File | Source | Why it is not here |
|---|---|---|
| `gas_sap_daily_icis.csv` | ICIS, gas System Average Price (SAP), daily | ICIS is a commercial price-reporting agency; its terms prohibit redistribution of its price assessments. |
| `carbon_ukets_futures_daily_investing.csv` | Investing.com, UK ETS carbon futures, daily | Investing.com's terms of use prohibit republication of its data. |

Expected schemas, so a substitute series can be dropped in:

- `gas_sap_daily_icis.csv` — two columns, `Date` (`YYYY-MM-DD`) and
  `SAP actual day`, one row per day, in the units the source publishes. A daily NBP or SAP series from any
  source you are licensed to use will work. Ofgem publishes an SAP series.
- `carbon_ukets_futures_daily_investing.csv` — Investing.com's standard export:
  `"Date","Price","Open","High","Low","Vol.","Change %"` with `Date` as
  `DD/MM/YYYY` and quoted numeric strings. Only `Date` and `Price`
  (EUR/tCO2) are read.

Both series are resampled from daily to hourly and lagged by one day in Stage 2,
so daily resolution is sufficient.
