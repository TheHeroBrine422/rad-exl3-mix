
### doc 1893 (chat, wildchat): 682 tokens, 583 gen rows
| checkpoint | KL / row (unhealed → checkpoint) | blows up | jump at position (rolling 32 gen-row mean ≥ 0.3 and ≥ 5× control) | KL / row before → after | token there (context) |
|---|---|---|---|---|---|
| 25A-ckpt-2e+05 | 0.0695 → 0.0400 | no | - | - | - |
| 25A-ckpt-5e+05 | 0.0695 → 1.6619 | **yes** | 240 of 682 | 0.044 → 2.188 | , pero manejable con una ⟦ estrategia⟧ |
| 25A-lr3e-4-ckpt-2e+05 | 0.0695 → 0.0584 | no | - | - | - |
| 25A-lr3e-4-ckpt-5e+05 | 0.0695 → 0.0572 | no | - | - | - |
| 25E-l1e-3-x1-ckpt-2e+05 | 0.0695 → 0.0443 | no | - | - | - |
| 25E-l1e-3-x1-ckpt-5e+05 | 0.0695 → 0.4990 | **yes** | 252 of 682 | 0.043 → 0.664 |  algunos pasos y consideraciones para ⟦ gestionar⟧ |
| 30A-ckpt-2e+05 | 0.1100 → 0.0856 | no | - | - | - |
| 30A-ckpt-5e+05 | 0.1100 → 0.0539 | no | - | - | - |
| 30A-lr3e-4-ckpt-2e+05 | 0.1100 → 0.0984 | no | - | - | - |
| 30A-lr3e-4-ckpt-5e+05 | 0.1100 → 0.0752 | no | - | - | - |
| 30E-l1e-3-x1-ckpt-2e+05 | 0.1100 → 0.0890 | no | - | - | - |
| 30E-l1e-3-x1-ckpt-5e+05 | 0.1100 → 0.0258 | no | - | - | - |
| 30E-l3e-4-x1-ckpt-2e+05 | 0.1100 → 0.0978 | no | - | - | - |
| 30E-l3e-4-x1-ckpt-5e+05 | 0.1100 → 0.0778 | no | - | - | - |
| 40A-ckpt-2e+05 | 0.0212 → 0.0157 | no | - | - | - |
| 40A-ckpt-5e+05 | 0.0212 → 0.0499 | no | - | - | - |
| 40A-lr3e-4-ckpt-2e+05 | 0.0212 → 0.0145 | no | - | - | - |
| 40A-lr3e-4-ckpt-5e+05 | 0.0212 → 0.0115 | no | - | - | - |
| 40E-l3e-4-x1-ckpt-2e+05 | 0.0212 → 0.0305 | no | - | - | - |
| 40E-l3e-4-x1-ckpt-5e+05 | 0.0212 → 0.0058 | no | - | - | - |

### doc 3820 (chat, wildchat): 1137 tokens, 957 gen rows
| checkpoint | KL / row (unhealed → checkpoint) | blows up | jump at position (rolling 32 gen-row mean ≥ 0.3 and ≥ 5× control) | KL / row before → after | token there (context) |
|---|---|---|---|---|---|
| 25A-ckpt-2e+05 | 0.5325 → 0.6272 | no | - | - | - |
| 25A-ckpt-5e+05 | 0.5325 → 0.1644 | no | - | - | - |
| 25A-lr3e-4-ckpt-2e+05 | 0.5325 → 0.5483 | no | - | - | - |
| 25A-lr3e-4-ckpt-5e+05 | 0.5325 → 0.1106 | no | - | - | - |
| 25E-l1e-3-x1-ckpt-2e+05 | 0.5325 → 0.6811 | no | - | - | - |
| 25E-l1e-3-x1-ckpt-5e+05 | 0.5325 → 0.1770 | no | - | - | - |
| 30A-ckpt-2e+05 | 0.0911 → 0.1133 | no | - | - | - |
| 30A-ckpt-5e+05 | 0.0911 → 0.1638 | no | - | - | - |
| 30A-lr3e-4-ckpt-2e+05 | 0.0911 → 0.0787 | no | - | - | - |
| 30A-lr3e-4-ckpt-5e+05 | 0.0911 → 0.0956 | no | - | - | - |
| 30E-l1e-3-x1-ckpt-2e+05 | 0.0911 → 0.1048 | no | - | - | - |
| 30E-l1e-3-x1-ckpt-5e+05 | 0.0911 → 0.1860 | no | - | - | - |
| 30E-l3e-4-x1-ckpt-2e+05 | 0.0911 → 0.0976 | no | - | - | - |
| 30E-l3e-4-x1-ckpt-5e+05 | 0.0911 → 0.0649 | no | - | - | - |
| 40A-ckpt-2e+05 | 0.0259 → 0.1143 | no | - | - | - |
| 40A-ckpt-5e+05 | 0.0259 → 1.7985 | **yes** | 202 of 1137 | 0.050 → 1.843 |  text into English step-by-step ⟦,⟧ |
| 40A-lr3e-4-ckpt-2e+05 | 0.0259 → 0.0452 | no | - | - | - |
| 40A-lr3e-4-ckpt-5e+05 | 0.0259 → 1.9953 | **yes** | 210 of 1137 | 0.018 → 2.064 |  I'll explain the choices made ⟦ during⟧ |
| 40E-l3e-4-x1-ckpt-2e+05 | 0.0259 → 0.0320 | no | - | - | - |
| 40E-l3e-4-x1-ckpt-5e+05 | 0.0259 → 1.8987 | **yes** | 221 of 1137 | 0.017 → 1.987 | \n\n### Original Text:\n ⟦ب⟧ |
