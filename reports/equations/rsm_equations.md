# Response Surface Methodology (RSM) Explicit Design Equations

Second-order polynomial formulations derived across raw material costs and span length inputs.

---

## Target: `Gir Dep (in)`
- **R² Score:** 0.6610
- **RMSE:** 1.3861
- **Equation:**
```text
Gir Dep (in) = +44.5074 +0.002748 * Concrete +22.369118 * Strand -4.760544 * Rebar +0.099547 * Span_ft +0.000011 * Concrete^2 -0.004663 * Concrete * Strand -0.002801 * Concrete * Rebar +0.000009 * Concrete * Span_ft -3.920456 * Strand^2 -0.149617 * Strand * Rebar -0.030436 * Strand * Span_ft +0.484014 * Rebar^2 +0.019657 * Rebar * Span_ft -0.000179 * Span_ft^2
```

## Target: `Lat Spac (ft)`
- **R² Score:** 0.8306
- **RMSE:** 0.3611
- **Equation:**
```text
Lat Spac (ft) = -4.9523 -0.001228 * Concrete +0.824492 * Strand -0.077151 * Rebar +0.187315 * Span_ft +0.000539 * Concrete * Strand +0.000501 * Concrete * Rebar -0.000002 * Concrete * Span_ft -0.217914 * Strand^2 +0.063843 * Strand * Rebar -0.003307 * Strand * Span_ft -0.141892 * Rebar^2 +0.002765 * Rebar * Span_ft -0.000750 * Span_ft^2
```

## Target: `No. of Gir`
- **R² Score:** 0.7847
- **RMSE:** 0.6213
- **Equation:**
```text
No. of Gir = +22.3940 +0.002901 * Concrete -1.447943 * Strand +0.201113 * Rebar -0.261090 * Span_ft -0.000747 * Concrete * Strand -0.001554 * Concrete * Rebar +0.000016 * Concrete * Span_ft +0.411889 * Strand^2 -0.066461 * Strand * Rebar +0.004396 * Strand * Span_ft +0.287385 * Rebar^2 -0.005417 * Rebar * Span_ft +0.001050 * Span_ft^2
```

## Target: `bot flange bot part depth (in)`
- **R² Score:** 0.1203
- **RMSE:** 1.4929
- **Equation:**
```text
bot flange bot part depth (in) = +24.0323 -0.058186 * Concrete -2.921621 * Strand -2.514897 * Rebar +0.051839 * Span_ft +0.000046 * Concrete^2 +0.006486 * Concrete * Strand +0.001850 * Concrete * Rebar -0.000024 * Concrete * Span_ft -0.644544 * Strand^2 -0.525141 * Strand * Rebar +0.020309 * Strand * Span_ft +0.673785 * Rebar^2 -0.008590 * Rebar * Span_ft -0.000161 * Span_ft^2
```

## Target: `bot flange bot part width (in)`
- **R² Score:** 0.6043
- **RMSE:** 5.7873
- **Equation:**
```text
bot flange bot part width (in) = -138.6298 +0.026408 * Concrete -19.847148 * Strand +25.923205 * Rebar +2.023460 * Span_ft -0.000098 * Concrete^2 +0.013797 * Concrete * Strand +0.011088 * Concrete * Rebar +0.000046 * Concrete * Span_ft +7.683385 * Strand^2 -1.547968 * Strand * Rebar -0.055343 * Strand * Span_ft -4.435635 * Rebar^2 -0.017589 * Rebar * Span_ft -0.006079 * Span_ft^2
```

## Target: `Number of strand per girder`
- **R² Score:** 0.9638
- **RMSE:** 4.1084
- **Equation:**
```text
Number of strand per girder = -34.4420 -0.135041 * Concrete -22.372465 * Strand -5.081294 * Rebar +1.659600 * Span_ft +0.000084 * Concrete^2 +0.008167 * Concrete * Strand +0.014053 * Concrete * Rebar +0.000021 * Concrete * Span_ft +3.770338 * Strand^2 +1.658940 * Strand * Rebar -0.020376 * Strand * Span_ft -1.173957 * Rebar^2 +0.012863 * Rebar * Span_ft -0.003324 * Span_ft^2
```

## Target: `Harp Pos (ft)`
- **R² Score:** 0.7656
- **RMSE:** 4.3430
- **Equation:**
```text
Harp Pos (ft) = +24.3191 -0.101973 * Concrete +4.769822 * Strand +25.886897 * Rebar -0.094973 * Span_ft +0.000079 * Concrete^2 -0.006320 * Concrete * Strand +0.014837 * Concrete * Rebar -0.000041 * Concrete * Span_ft +3.631444 * Strand^2 -4.130234 * Strand * Rebar -0.027149 * Strand * Span_ft -3.278885 * Rebar^2 -0.070170 * Rebar * Span_ft +0.002237 * Span_ft^2
```
