# Dataset and model input contract

## Dataset status

The checked-in CSV has 7,032 rows and 23 columns. The target `Churn` is
encoded as 0 (retained) or 1 (churned); positive-class prevalence is 26.6%.
There are 1,520 missing values, all in `InternetService`. The six internet
add-on fields are 0 for those rows, but `PhoneService` is 1 for those rows
only as an observed dataset pattern. Phone and internet service are separate
concepts and the application does not enforce that correlation.

The upstream source, licence, observation date, and exact future outcome
horizon have not been verified. The missing `InternetService` pattern is
therefore an unresolved source-data mapping. The UI's “No internet service”
choice maps to the missing category learned by this model; validate this
interpretation before using predictions operationally.

## Model features

| Feature | Type | Accepted model representation |
| --- | --- | --- |
| `tenure` | Numeric | Non-negative whole months |
| `MonthlyCharges` | Numeric | Finite, non-negative amount |
| `TotalCharges` | Numeric | Finite, non-negative amount |
| `Partner`, `Dependents`, `PhoneService`, `MultipleLines`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`, `PaperlessBilling` | Binary | 0 or 1 |
| `InternetService` | Categorical | `DSL`, `Fiber-optic`, or the training missing-value category |
| `Contract` | Categorical | `Month-to-month`, `One-year`, `Two-year` |
| `PaymentMethod` | Categorical | `Bank-transfer`, `Credit-card`, `Electronic-check`, `Mailed-check` |

The current development split ranges are `tenure` 1–72 months,
`MonthlyCharges` 18.40–118.65, and `TotalCharges` 18.80–8,684.80. They are
recorded in model metadata. Values outside these observed ranges are scored
but explicitly flagged as extrapolation; the range is not a universal
business limit.

The model excludes `gender`, `SeniorCitizen`, `Index`, `Customer type`, and
`Custom value`. The last three include identifiers or values derived from
other columns. Excluding protected attributes does not establish fairness;
proxies and subgroup error rates still require review.

## Service consistency rules

- `PhoneService = 0` requires `MultipleLines = 0`.
- Missing/no `InternetService` requires all six internet add-on flags to be 0.
- Phone service may be 0 or 1 when internet service is unavailable.
- Individual no-internet add-on answers normalize to 0.
- The individual form derives unavailable dependent services from the selected
  parent service and tells the user when it does so.

Batch prediction accepts the model representation above. Binary columns must
be 0/1. `InternetService` additionally accepts blank, `No`, `No internet
service`, and `Not recorded` as aliases for the learned missing-value
category. Unknown categories and contradictory service combinations are
rejected rather than silently repaired.

## Non-model fields

`Churn` is the evaluation target and must never be included as a feature.
`gender`, `SeniorCitizen`, `Index`, `Customer type`, and `Custom value` are
not model inputs. Extra batch columns are preserved for user reference and
must not contain unnecessary personal or sensitive information.
