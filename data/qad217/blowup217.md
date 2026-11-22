
### doc 1893 (chat, wildchat): 682 tokens, 583 gen rows
| checkpoint | KL / row (unhealed → checkpoint) | blows up | jump at position (rolling 32 gen-row mean ≥ 0.3 and ≥ 5× control) | KL / row before → after | token there (context) |
|---|---|---|---|---|---|
| 25A-ckpt-2e+05 | 0.0695 → 0.0400 | no | - | - | - |
| 25A-ckpt-5e+05 | 0.0695 → 1.6619 | **yes** | 240 of 682 | 0.044 → 2.188 | , pero manejable con una ⟦ estrategia⟧ |
| 25A-lr3e-4-ckpt-2e+05 | 0.0695 → 0.0584 | no | - | - | - |
| 25A-lr3e-4-ckpt-5e+05 | 0.0695 → 0.0572 | no | - | - | - |
| 25A-lr3e-4-ext1M-ckpt-1e+06 | 0.0695 → 0.2577 | no | - | - | - |
| 25E-l1e-3-x0.3-ckpt-2e+05 | 0.0695 → 0.0382 | no | - | - | - |
| 25E-l1e-3-x0.3-ckpt-5e+05 | 0.0695 → 1.5404 | **yes** | 241 of 682 | 0.045 → 2.031 |  pero manejable con una estrategia ⟦ apropi⟧ |
| 25E-l1e-3-x1-ckpt-2e+05 | 0.0695 → 0.0443 | no | - | - | - |
| 25E-l1e-3-x1-ckpt-5e+05 | 0.0695 → 0.4990 | **yes** | 252 of 682 | 0.043 → 0.664 |  algunos pasos y consideraciones para ⟦ gestionar⟧ |
| 25E-l1e-3-x3-ckpt-2e+05 | 0.0695 → 0.0498 | no | - | - | - |
| 25E-l1e-3-x3-ckpt-5e+05 | 0.0695 → 0.1601 | no | - | - | - |
| 25E-l3e-4-x0.3-ckpt-2e+05 | 0.0695 → 0.0565 | no | - | - | - |
| 25E-l3e-4-x0.3-ckpt-5e+05 | 0.0695 → 0.0551 | no | - | - | - |
| 25E-l3e-4-x1-ckpt-2e+05 | 0.0695 → 0.0608 | no | - | - | - |
| 25E-l3e-4-x1-ckpt-5e+05 | 0.0695 → 0.0450 | no | - | - | - |
| 30A-ckpt-2e+05 | 0.1100 → 0.0856 | no | - | - | - |
| 30A-ckpt-5e+05 | 0.1100 → 0.0539 | no | - | - | - |
| 30A-lr3e-4-ckpt-2e+05 | 0.1100 → 0.0984 | no | - | - | - |
| 30A-lr3e-4-ckpt-5e+05 | 0.1100 → 0.0752 | no | - | - | - |
| 30A-lr3e-4-ext1M-ckpt-1e+06 | 0.1100 → 0.1408 | no | - | - | - |
| 30E-l1e-3-x0.3-ckpt-2e+05 | 0.1100 → 0.0945 | no | - | - | - |
| 30E-l1e-3-x0.3-ckpt-5e+05 | 0.1100 → 0.0372 | no | - | - | - |
| 30E-l1e-3-x1-ckpt-2e+05 | 0.1100 → 0.0890 | no | - | - | - |
| 30E-l1e-3-x1-ckpt-5e+05 | 0.1100 → 0.0258 | no | - | - | - |
| 30E-l1e-3-x3-ckpt-2e+05 | 0.1100 → 0.1056 | no | - | - | - |
| 30E-l1e-3-x3-ckpt-5e+05 | 0.1100 → 0.0394 | no | - | - | - |
| 30E-l3e-4-x0.3-ckpt-2e+05 | 0.1100 → 0.0937 | no | - | - | - |
| 30E-l3e-4-x0.3-ckpt-5e+05 | 0.1100 → 0.0859 | no | - | - | - |
| 30E-l3e-4-x1-ckpt-2e+05 | 0.1100 → 0.0978 | no | - | - | - |
| 30E-l3e-4-x1-ckpt-5e+05 | 0.1100 → 0.0778 | no | - | - | - |
| 30E-l3e-4-x3-ckpt-2e+05 | 0.1100 → 0.1033 | no | - | - | - |
| 30E-l3e-4-x3-ckpt-5e+05 | 0.1100 → 0.1067 | no | - | - | - |
| 40A-ckpt-2e+05 | 0.0212 → 0.0157 | no | - | - | - |
| 40A-ckpt-5e+05 | 0.0212 → 0.0499 | no | - | - | - |
| 40A-lr3e-4-ckpt-2e+05 | 0.0212 → 0.0145 | no | - | - | - |
| 40A-lr3e-4-ckpt-5e+05 | 0.0212 → 0.0115 | no | - | - | - |
| 40E-l1e-3-x0.3-ckpt-2e+05 | 0.0212 → 0.0187 | no | - | - | - |
| 40E-l1e-3-x0.3-ckpt-5e+05 | 0.0212 → 0.0669 | no | - | - | - |
| 40E-l1e-3-x1-ckpt-2e+05 | 0.0212 → 0.0197 | no | - | - | - |
| 40E-l1e-3-x1-ckpt-5e+05 | 0.0212 → 0.0454 | no | - | - | - |
| 40E-l1e-3-x3-ckpt-2e+05 | 0.0212 → 0.0153 | no | - | - | - |
| 40E-l1e-3-x3-ckpt-5e+05 | 0.0212 → 0.0212 | no | - | - | - |
| 40E-l3e-4-x0.3-ckpt-2e+05 | 0.0212 → 0.0211 | no | - | - | - |
| 40E-l3e-4-x0.3-ckpt-5e+05 | 0.0212 → 0.0064 | no | - | - | - |
| 40E-l3e-4-x1-ckpt-2e+05 | 0.0212 → 0.0305 | no | - | - | - |
| 40E-l3e-4-x1-ckpt-5e+05 | 0.0212 → 0.0058 | no | - | - | - |

### doc 3820 (chat, wildchat): 1137 tokens, 957 gen rows
| checkpoint | KL / row (unhealed → checkpoint) | blows up | jump at position (rolling 32 gen-row mean ≥ 0.3 and ≥ 5× control) | KL / row before → after | token there (context) |
|---|---|---|---|---|---|
| 25A-ckpt-2e+05 | 0.5325 → 0.6272 | no | - | - | - |
| 25A-ckpt-5e+05 | 0.5325 → 0.1644 | no | - | - | - |
| 25A-lr3e-4-ckpt-2e+05 | 0.5325 → 0.5483 | no | - | - | - |
| 25A-lr3e-4-ckpt-5e+05 | 0.5325 → 0.1106 | no | - | - | - |
| 25A-lr3e-4-ext1M-ckpt-1e+06 | 0.5325 → 0.1954 | no | - | - | - |
| 25E-l1e-3-x0.3-ckpt-2e+05 | 0.5325 → 0.6210 | no | - | - | - |
| 25E-l1e-3-x0.3-ckpt-5e+05 | 0.5325 → 0.1882 | no | - | - | - |
| 25E-l1e-3-x1-ckpt-2e+05 | 0.5325 → 0.6811 | no | - | - | - |
| 25E-l1e-3-x1-ckpt-5e+05 | 0.5325 → 0.1770 | no | - | - | - |
| 25E-l1e-3-x3-ckpt-2e+05 | 0.5325 → 0.6209 | no | - | - | - |
| 25E-l1e-3-x3-ckpt-5e+05 | 0.5325 → 0.1452 | no | - | - | - |
| 25E-l3e-4-x0.3-ckpt-2e+05 | 0.5325 → 0.6440 | no | - | - | - |
| 25E-l3e-4-x0.3-ckpt-5e+05 | 0.5325 → 0.0947 | no | - | - | - |
| 25E-l3e-4-x1-ckpt-2e+05 | 0.5325 → 0.6026 | no | - | - | - |
| 25E-l3e-4-x1-ckpt-5e+05 | 0.5325 → 0.0806 | no | - | - | - |
| 30A-ckpt-2e+05 | 0.0911 → 0.1133 | no | - | - | - |
| 30A-ckpt-5e+05 | 0.0911 → 0.1638 | no | - | - | - |
| 30A-lr3e-4-ckpt-2e+05 | 0.0911 → 0.0787 | no | - | - | - |
| 30A-lr3e-4-ckpt-5e+05 | 0.0911 → 0.0956 | no | - | - | - |
| 30A-lr3e-4-ext1M-ckpt-1e+06 | 0.0911 → 0.1517 | no | - | - | - |
| 30E-l1e-3-x0.3-ckpt-2e+05 | 0.0911 → 0.0890 | no | - | - | - |
| 30E-l1e-3-x0.3-ckpt-5e+05 | 0.0911 → 0.2147 | no | - | - | - |
| 30E-l1e-3-x1-ckpt-2e+05 | 0.0911 → 0.1048 | no | - | - | - |
| 30E-l1e-3-x1-ckpt-5e+05 | 0.0911 → 0.1860 | no | - | - | - |
| 30E-l1e-3-x3-ckpt-2e+05 | 0.0911 → 0.1001 | no | - | - | - |
| 30E-l1e-3-x3-ckpt-5e+05 | 0.0911 → 0.1563 | no | - | - | - |
| 30E-l3e-4-x0.3-ckpt-2e+05 | 0.0911 → 0.1032 | no | - | - | - |
| 30E-l3e-4-x0.3-ckpt-5e+05 | 0.0911 → 0.0499 | no | - | - | - |
| 30E-l3e-4-x1-ckpt-2e+05 | 0.0911 → 0.0976 | no | - | - | - |
| 30E-l3e-4-x1-ckpt-5e+05 | 0.0911 → 0.0649 | no | - | - | - |
| 30E-l3e-4-x3-ckpt-2e+05 | 0.0911 → 0.1109 | no | - | - | - |
| 30E-l3e-4-x3-ckpt-5e+05 | 0.0911 → 0.1938 | no | - | - | - |
| 40A-ckpt-2e+05 | 0.0259 → 0.1143 | no | - | - | - |
| 40A-ckpt-5e+05 | 0.0259 → 1.7985 | **yes** | 202 of 1137 | 0.050 → 1.843 |  text into English step-by-step ⟦,⟧ |
| 40A-lr3e-4-ckpt-2e+05 | 0.0259 → 0.0452 | no | - | - | - |
| 40A-lr3e-4-ckpt-5e+05 | 0.0259 → 1.9953 | **yes** | 210 of 1137 | 0.018 → 2.064 |  I'll explain the choices made ⟦ during⟧ |
| 40E-l1e-3-x0.3-ckpt-2e+05 | 0.0259 → 0.0603 | no | - | - | - |
| 40E-l1e-3-x0.3-ckpt-5e+05 | 0.0259 → 2.0947 | **yes** | 197 of 1137 | 0.189 → 2.133 |  of the Tunisian dialect text ⟦ into⟧ |
| 40E-l1e-3-x1-ckpt-2e+05 | 0.0259 → 0.0426 | no | - | - | - |
| 40E-l1e-3-x1-ckpt-5e+05 | 0.0259 → 2.1563 | **yes** | 200 of 1137 | 0.078 → 2.205 | ian dialect text into English step ⟦-by⟧ |
| 40E-l1e-3-x3-ckpt-2e+05 | 0.0259 → 0.0336 | no | - | - | - |
| 40E-l1e-3-x3-ckpt-5e+05 | 0.0259 → 2.4012 | **yes** | 203 of 1137 | 0.050 → 2.464 |  into English step-by-step, ⟦ and⟧ |
| 40E-l3e-4-x0.3-ckpt-2e+05 | 0.0259 → 0.0427 | no | - | - | - |
| 40E-l3e-4-x0.3-ckpt-5e+05 | 0.0259 → 2.0540 | **yes** | 214 of 1137 | 0.019 → 2.134 |  choices made during the translation process ⟦.⟧ |
| 40E-l3e-4-x1-ckpt-2e+05 | 0.0259 → 0.0320 | no | - | - | - |
| 40E-l3e-4-x1-ckpt-5e+05 | 0.0259 → 1.8987 | **yes** | 221 of 1137 | 0.017 → 1.987 | \n\n### Original Text:\n ⟦ب⟧ |
