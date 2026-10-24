# Data

## `hydrochem_At_LLt.csv`

Hydrochemical compositions of surface water in the Llobregat river basin
(NE Spain), from the R package **compositions** (version 2.0-9, file
`data/Hydrochem.txt`, GPL ≥ 2; Otero, Tolosana-Delgado, Soler,
Pawlowsky-Glahn & Canals 2005, *Water Research* 39(7):1404–1414). The
original documentation is copied to `Hydrochem.Rd.source`.

This is the subset analysed by Yu & Huang (2024, EJS, Sec. 1.8) and Yu &
Huang (2025, CSDA, Sec. 7.1):

| column | meaning |
|---|---|
| `group` | `At` = tributaries of the Anoia (n = 67), `LLt` = tributaries of the lower Llobregat (n = 43) |
| `x` | covariate: 0 for `At`, 1 for `LLt` |
| `K`, `Na`, `Ca`, `Mg` | molarities of K⁺, Na⁺, Ca²⁺, Mg²⁺ as in the source file |
| `y1..y4` | response on S³: component-wise square root of the closed composition `(K, Na, Ca, Mg)/(K+Na+Ca+Mg)` |

Generated with:

```sh
awk 'NR>1{loc=$4; gsub(/"/,"",loc); if(loc!="At" && loc!="LLt") next;
     na=$8; k=$9; mg=$10; ca=$11; s=k+na+ca+mg; x=(loc=="LLt")?1:0;
     printf "%s,%d,%.10g,%.10g,%.10g,%.10g,%.17g,%.17g,%.17g,%.17g\n",
       loc, x, k, na, ca, mg, sqrt(k/s), sqrt(na/s), sqrt(ca/s), sqrt(mg/s)}' Hydrochem.txt
```

Group mean compositions (0.037, 0.274, 0.516, 0.173) for `At` and
(0.054, 0.371, 0.401, 0.174) for `LLt` agree with the papers'
(0.04, 0.28, 0.51, 0.17) and (0.06, 0.37, 0.40, 0.17).
