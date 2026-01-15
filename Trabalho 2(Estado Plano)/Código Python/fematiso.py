import numpy as np

def fematiso(iopt:  int, elastic: float, poisson: float) -> np.ndarray:
    """
    Determina a equação constitutiva para material isotrópico.
    
    Parâmetros: 
    -----------
    iopt : int
        Tipo de análise: 
        1 - Estado plano de tensões
        2 - Estado plano de deformações
        3 - Análise axissimétrica
        4 - Análise tridimensional
    elastic : float
        Módulo de elasticidade (E)
    poisson : float
        Coeficiente de Poisson (ν)
    
    Retorna:
    --------
    matmtrx : np.ndarray
        Matriz constitutiva [D]
    """
    
    if iopt == 1:  # Estado plano de tensões
        coef = elastic / (1 - poisson**2)
        matmtrx = coef * np.array([
            [1,       poisson, 0                  ],
            [poisson, 1,       0                  ],
            [0,       0,       (1 - poisson) / 2  ]
        ])
    
    elif iopt == 2:  # Estado plano de deformações
        coef = elastic / ((1 + poisson) * (1 - 2*poisson))
        matmtrx = coef * np.array([
            [(1 - poisson), poisson,       0                    ],
            [poisson,       (1 - poisson), 0                    ],
            [0,             0,             (1 - 2*poisson) / 2  ]
        ])
    
    elif iopt == 3:  # Axissimétrico
        coef = elastic / ((1 + poisson) * (1 - 2*poisson))
        matmtrx = coef * np.array([
            [(1 - poisson), poisson,       poisson,       0                    ],
            [poisson,       (1 - poisson), poisson,       0                    ],
            [poisson,       poisson,       (1 - poisson), 0                    ],
            [0,             0,             0,             (1 - 2*poisson) / 2  ]
        ])
    
    else:  # Tridimensional (iopt == 4)
        coef = elastic / ((1 + poisson) * (1 - 2*poisson))
        matmtrx = coef * np.array([
            [(1-poisson), poisson,     poisson,     0,                  0,                  0                  ],
            [poisson,     (1-poisson), poisson,     0,                  0,                  0                  ],
            [poisson,     poisson,     (1-poisson), 0,                  0,                  0                  ],
            [0,           0,           0,           (1-2*poisson)/2,    0,                  0                  ],
            [0,           0,           0,           0,                  (1-2*poisson)/2,    0                  ],
            [0,           0,           0,           0,                  0,                  (1-2*poisson)/2    ]
        ])
    
    return matmtrx