# Response Surface Methodology (RSM) Explicit Design Equations

Second-order polynomial formulations derived across raw material costs and span length inputs.

---

## Target: `Gir Dep (in)`
- **R² Score:** 0.1941
- **RMSE:** 3.9444
- **Equation:**
```text
Gir Dep (in) = +44.5074 +0.002748 * Concrete +22.369118 * Strand -4.760544 * Rebar +0.099547 * Span_ft +0.000011 * Concrete^2 -0.004663 * Concrete * Strand -0.002801 * Concrete * Rebar +0.000009 * Concrete * Span_ft -3.920456 * Strand^2 -0.149617 * Strand * Rebar -0.030436 * Strand * Span_ft +0.484014 * Rebar^2 +0.019657 * Rebar * Span_ft -0.000179 * Span_ft^2
```

## Target: `Lat Spac (ft)`
- **R² Score:** 0.5274
- **RMSE:** 0.7568
- **Equation:**
```text
Lat Spac (ft) = -4.9523 -0.001228 * Concrete +0.824492 * Strand -0.077151 * Rebar +0.187315 * Span_ft +0.000539 * Concrete * Strand +0.000501 * Concrete * Rebar -0.000002 * Concrete * Span_ft -0.217914 * Strand^2 +0.063843 * Strand * Rebar -0.003307 * Strand * Span_ft -0.141892 * Rebar^2 +0.002765 * Rebar * Span_ft -0.000750 * Span_ft^2
```

## Target: `No. of Gir`
- **R² Score:** 0.5628
- **RMSE:** 1.0727
- **Equation:**
```text
No. of Gir = +23.5064 +0.003969 * Concrete -0.764795 * Strand -0.062802 * Rebar -0.285145 * Span_ft +0.000001 * Concrete^2 -0.000935 * Concrete * Strand -0.001346 * Concrete * Rebar +0.000003 * Concrete * Span_ft +0.229808 * Strand^2 -0.132333 * Strand * Rebar +0.005558 * Strand * Span_ft +0.297839 * Rebar^2 -0.004194 * Rebar * Span_ft +0.001140 * Span_ft^2
```

## Target: `bot flange bot part depth (in)`
- **R² Score:** 0.0323
- **RMSE:** 3.0204
- **Equation:**
```text
bot flange bot part depth (in) = +24.0323 -0.058186 * Concrete -2.921621 * Strand -2.514897 * Rebar +0.051839 * Span_ft +0.000046 * Concrete^2 +0.006486 * Concrete * Strand +0.001850 * Concrete * Rebar -0.000024 * Concrete * Span_ft -0.644544 * Strand^2 -0.525141 * Strand * Rebar +0.020309 * Strand * Span_ft +0.673785 * Rebar^2 -0.008590 * Rebar * Span_ft -0.000161 * Span_ft^2
```

## Target: `bot flange bot part width (in)`
- **R² Score:** 0.2566
- **RMSE:** 12.1745
- **Equation:**
```text
bot flange bot part width (in) = -138.6298 +0.026408 * Concrete -19.847148 * Strand +25.923205 * Rebar +2.023460 * Span_ft -0.000098 * Concrete^2 +0.013797 * Concrete * Strand +0.011088 * Concrete * Rebar +0.000046 * Concrete * Span_ft +7.683385 * Strand^2 -1.547968 * Strand * Rebar -0.055343 * Strand * Span_ft -4.435635 * Rebar^2 -0.017589 * Rebar * Span_ft -0.006079 * Span_ft^2
```

## Target: `Number of strand per girder`
- **R² Score:** 0.8530
- **RMSE:** 8.7831
- **Equation:**
```text
Number of strand per girder = -38.2086 -0.108069 * Concrete -24.262612 * Strand -4.315810 * Rebar +1.623666 * Span_ft +0.000061 * Concrete^2 +0.007040 * Concrete * Strand +0.013147 * Concrete * Rebar +0.000026 * Concrete * Span_ft +4.286573 * Strand^2 +1.927072 * Strand * Rebar -0.021130 * Strand * Span_ft -1.201516 * Rebar^2 +0.008894 * Rebar * Span_ft -0.003164 * Span_ft^2
```

## Target: `Harp Pos (ft)`
- **R² Score:** 0.3481
- **RMSE:** 10.7419
- **Equation:**
```text
Harp Pos (ft) = +24.3191 -0.101973 * Concrete +4.769822 * Strand +25.886897 * Rebar -0.094973 * Span_ft +0.000079 * Concrete^2 -0.006320 * Concrete * Strand +0.014837 * Concrete * Rebar -0.000041 * Concrete * Span_ft +3.631444 * Strand^2 -4.130234 * Strand * Rebar -0.027149 * Strand * Span_ft -3.278885 * Rebar^2 -0.070170 * Rebar * Span_ft +0.002237 * Span_ft^2
```
