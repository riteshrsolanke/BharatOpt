NAME          LOCAL_MODEL
ROWS
 N  OBJ
 L  CDU_Throughput
 L  Desulfurization
 G  Gasoline_Yield
 L  Slurry_Limit
COLUMNS
    Arab_Light_bbl OBJ        24.5
    Arab_Light_bbl CDU_Throughput 1.0
    Arab_Light_bbl Desulfurization 1.8
    Arab_Light_bbl Gasoline_Yield 0.28
    Arab_Light_bbl Slurry_Limit 0.12
    Bonny_Light_bbl OBJ        28.6
    Bonny_Light_bbl CDU_Throughput 1.0
    Bonny_Light_bbl Desulfurization 0.4
    Bonny_Light_bbl Gasoline_Yield 0.32
    Bonny_Light_bbl Slurry_Limit 0.05
    Brent_Blend_bbl OBJ        31.2
    Brent_Blend_bbl CDU_Throughput 1.0
    Brent_Blend_bbl Desulfurization 0.9
    Brent_Blend_bbl Gasoline_Yield 0.35
    Brent_Blend_bbl Slurry_Limit 0.08
    Maya_Heavy_bbl OBJ        18.75
    Maya_Heavy_bbl CDU_Throughput 1.0
    Maya_Heavy_bbl Desulfurization 3.4
    Maya_Heavy_bbl Gasoline_Yield 0.15
    Maya_Heavy_bbl Slurry_Limit 0.38
RHS
    RHS1      CDU_Throughput 150000.0
    RHS1      Desulfurization 220000.0
    RHS1      Gasoline_Yield 35000.0
    RHS1      Slurry_Limit 25000.0
BOUNDS
 LO BND       Arab_Light_bbl 0.0
 LO BND       Bonny_Light_bbl 0.0
 LO BND       Brent_Blend_bbl 0.0
 LO BND       Maya_Heavy_bbl 0.0
ENDATA