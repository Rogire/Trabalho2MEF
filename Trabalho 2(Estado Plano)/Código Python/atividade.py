import numpy as np
class ElementosFinitos:
    def __init__(self, Tipo_Estado: str, Mod_Elasticidade: float, poisson: float, Num_elementos:int,Num_nos:int):
        self.Tipo_Estado = Tipo_Estado
        self.Mod_Elasticidade = Mod_Elasticidade
        self.poisson = poisson
        self.Num_el = Num_elementos
        #considerando apenas elementos triangulares
        self.TotNodes = Num_nos


    def Calc_Mat_B(self,numEl: int, dNdx: np.ndarray, dNdy: np.ndarray) -> np.ndarray:
        """
        Calcula a matriz cinemática B, que relaciona deformações e deslocamentos
        para sólidos bidimensionais.
        """
        B = np.zeros((3, 2 * numEl))
        
        for i in range(numEl):
            i1 = 2 * i
            i2 = i1 + 1
            
            # ∂u/∂x (deformação εxx)
            B[0, i1] = dNdx[i]
            
            # ∂v/∂y (deformação εyy)
            B[1, i2] = dNdy[i]
            
            #∂u/∂y + ∂v/∂x (deformação γxy)
            B[2, i1] = dNdy[i]
            B[2, i2] = dNdx[i]
        
        return B

    def calc_IndexUV(self,index_nos_El: np.ndarray, Num_nos_el: int, Num_gLib_no: int) -> np.ndarray:
        """
        Calcula o index dos graus de liberdade (u,v) do elemento no sistema retorna em um vetor
        
        Exemplo:
        --------
        Para um elemento triangular (Num_nos_el=3) com nós [1, 3, 2] e Num_gLib_no=2:
        >>> index_nos_El = np.array([1, 3, 2])
        >>> index = feeldof(index_nos_El, 3, 2)
        >>> print(index)  # [1, 2, 5, 6, 3, 4] = [u1,v1, u3,v3, u2,v2]
        """
        
        numGlib = Num_nos_el * Num_gLib_no
        index = np.zeros(numGlib, dtype=int)
        k = 0
        
        for i in range(Num_nos_el):
            #-1 porque as contagens dos nós começa em 1
            start = (index_nos_El[i] - 1) * Num_gLib_no
            for j in range(Num_gLib_no):
                index[k] = start + j + 1
                k += 1
        
        return index
    
    def Mat_Rigidez(self,Mat_Global: np.ndarray, Mat_El: np.ndarray, index: np.ndarray) -> np.ndarray:
        #Montagem (assembly) das matrizes de rigidez dos elementos na matriz global.

        Glib_El = len(index)
        
        for i in range(Glib_El):
            #-1 porque as contagens dos nós começa em 1
            ii = index[i] - 1  
            for j in range(Glib_El):
                jj = index[j] - 1  
                Mat_Global[ii, jj] = Mat_Global[ii, jj] + Mat_El[i, j]
        
        return Mat_Global

    def CondCont(self, MatR_Global: np.ndarray, F: np.ndarray, indCC: np. ndarray, valCondCont: np. ndarray) -> tuple:
        """
        Aplica condições de contorno(CC) ao sistema [K]{u}={F}, escalona a matriz 
        de rigidez zerando a linha correspondente ao Grau de liberdade restringido,
        coloca 1 na diagonal e o valor prescrito no vetor de forças.

        Exemplo:
        Para prescrever u=0 nos Graus de liberdade 1 e 2 (u1,v1), e u=0. 5 no Grau de liberdade 5 (u3):
        >>> indCC = np.array([1, 2, 5])
        >>> valCondCont = np.array([0.0, 0.0, 0.5])
        >>> MatR_Global, F = CondCont(MatR_Global, F, indCC, valCondCont)
        """
        
        numCC = len(indCC)        
        sdof = MatR_Global.shape[0]  
        
        for i in range(numCC):
            #-1 porque as contagens dos nós começa em 1
            c = indCC[i] - 1  
            MatR_Global[c, : ] = 0
            MatR_Global[c, c] = 1
            F[c] = valCondCont[i]
        
        return MatR_Global, F
    
    def Rel_TenDef(self) -> np.ndarray:
        """
        Determina a equação constitutiva para material isotrópico.
        """
        
        if self.Tipo_Estado == "EPT":  # Estado plano de tensões
            coef = self.Mod_Elasticidade / (1 - self.poisson**2)
            res = coef * np.array([
                [1,       self.poisson, 0                  ],
                [self.poisson, 1,       0                  ],
                [0,       0,       (1 - self.poisson) / 2  ]
            ])
        
        elif self.Tipo_Estado == "EPD":  # Estado plano de deformações
            coef = self.Mod_Elasticidade / ((1 + self.poisson) * (1 - 2*self.poisson))
            res = coef * np.array([
                [(1 - self.poisson), self.poisson,       0                    ],
                [self.poisson,       (1 - self.poisson), 0                    ],
                [0,             0,             (1 - 2*self.poisson) / 2  ]
            ])
        else:
            res = np.array([])
        
        return res

    def CalculaEPT(self,F_desc,gcoord,nodes,bcdof,bcval):
        if self.Tipo_Estado != "EPT":
            print("Nao eh estado plano de tensao")
            return 

        
        GlibNo = 2       
        NumNosEl = 3           
        TotGlib = self.TotNodes * GlibNo        
        GlibEl = NumNosEl * GlibNo  # degrees of freedom per element
        
        F = np.zeros((TotGlib, 1))  # system force vector
        MatR_Global = np.zeros((TotGlib, TotGlib))  # system matrix
        disp = np.zeros((TotGlib, 1))  # system displacement vector
        eldisp = np.zeros((GlibEl, 1))  # element displacement vector
        stress = np.zeros((self.Num_el, 3))  # matrix containing stress components
        strain = np.zeros((self.Num_el, 3))  # matrix containing strain component
        C_sigma = np.zeros((3, 3))  # constitutive matrix
        index_Glib = np.zeros(GlibEl, dtype=int)  # index vector
        kinmtx = np.zeros((3, GlibEl))  # kinematic matrix


        # force vector
        # Converting from 1-based to 0-based indexing: 17 -> 16, 19 -> 18
        for i in F_desc:
            F[(i["no"]*GlibNo)+i["glib"] -1] = i["forca"]

        #F[16, 0] = 500  # force applied at node 9 in x-axis
        #F[18, 0] = 500  # force applied at node 10 in x-axis

        # compute element matrices and vectors, and assemble
        C_sigma = self.Rel_TenDef()  # constitutive matrix

        # DEBUG
        print("C_sigma =")
        print(C_sigma)

        for iel in range(self.Num_el):  # loop for the total number of elements
            nd = np.zeros(3, dtype=int)
            nd[0] = nodes[iel, 0]  # 1st connected node for (iel)-th element
            nd[1] = nodes[iel, 1]  # 2nd connected node for (iel)-th element
            nd[2] = nodes[iel, 2]  # 3rd connected node for (iel)-th element
            
            x1 = gcoord[nd[0], 0]
            y1 = gcoord[nd[0], 1]  # coord values of 1st node
            x2 = gcoord[nd[1], 0]
            y2 = gcoord[nd[1], 1]  # coord values of 2nd node
            x3 = gcoord[nd[2], 0]
            y3 = gcoord[nd[2], 1]  # coord values of 3rd node
            
            index_Glib = self.calc_IndexUV(nd, NumNosEl, GlibNo)  # extract system dofs for the element
            
            # find the derivatives of shape functions
            f1 = x2*y3 - x3*y2
            f2 = x3*y1 - x1*y3
            f3 = x1*y2 - x2*y1
            b1 = y2 - y3
            b2 = y3 - y1
            b3 = y1 - y2
            c1 = x3 - x2
            c2 = x1 - x3
            c3 = x2 - x1

            area = 0.5 * (f1+f2+f3)  # det da matriz da area
            area2 = 2*area
            dNdx = (1/area2) * np.array([b1, b2, b3])  # derivada de N em relação a x
            dNdy = (1/area2) * np.array([c1, c2, c3])  # derivada de N em relação a y
            
            B_mat = self.Calc_Mat_B(NumNosEl, dNdx, dNdy)  # kinematic matrix
            
            print("Matriz B.T: \n", B_mat.T)
            print("Matriz Csigma: \n", C_sigma)
            print("area: ", area)
            print("B*area: \n", B_mat*area)


            k = B_mat.T @ C_sigma @ B_mat * area  # element stiffness matrix
            
            print("Matriz k: \n",k)

            MatR_Global = self.Mat_Rigidez(MatR_Global, k, index_Glib)  
            # assemble element matrices

        # apply boundary conditions
        MatR_Global, F = self.CondCont(MatR_Global, F, bcdof, bcval)

        # DEBUG
        np.set_printoptions(linewidth=200, suppress=False, precision=4)

        print('System Force Vector:')
        print(F)

        # solve the matrix equation
        # [K]{u} = {F}
        print("Matriz Global: \n", MatR_Global)
        print("F: ", F)

        disp = np.linalg.solve(MatR_Global, F)

        # element stress computation (post computation)
        for ielp in range(self.Num_el):  # loop for the total number of elements
            nd = np.zeros(3, dtype=int)
            nd[0] = nodes[ielp, 0]  # 1st connected node for (iel)-th element
            nd[1] = nodes[ielp, 1]  # 2nd connected node for (iel)-th element
            nd[2] = nodes[ielp, 2]  # 3rd connected node for (iel)-th element
            
            x1 = gcoord[nd[0], 0]
            y1 = gcoord[nd[0], 1]  # coord values of 1st node
            x2 = gcoord[nd[1], 0]
            y2 = gcoord[nd[1], 1]  # coord values of 2nd node
            x3 = gcoord[nd[2], 0]
            y3 = gcoord[nd[2], 1]  # coord values of 3rd node
            
            index_Glib = self.calc_IndexUV(nd, NumNosEl, GlibNo)  # extract system dofs for the element
            
            # extract element displacement vector
            for i in range(GlibEl):
                eldisp[i, 0] = disp[index_Glib[i], 0]
            
            area = 0.5 * (x1*y2 + x2*y3 + x3*y1 - x1*y3 - x2*y1 - x3*y2)  # area of triangle
            area2 = area * 2
            dNdx = (1/area2) * np.array([y2-y3, y3-y1, y1-y2])  # derivatives w.r.t. x
            dNdy = (1/area2) * np.array([x3-x2, x1-x3, x2-x1])  # derivatives w.r.t. y
            
            B_mat = self.Calc_Mat_B(NumNosEl, dNdx, dNdy)  # kinematic matrix
            
            # DEBUG: force double precision
            eldisp = eldisp.astype(np.float64)
            B_mat = B_mat.astype(np.float64)
            
            estrain = B_mat @ eldisp  # compute strains
            estress = C_sigma @ estrain  # compute stresses
            
            for i in range(3):
                strain[ielp, i] = estrain[i, 0]  # store for each element
                stress[ielp, i] = estress[i, 0]  # store for each element
        
        # print fem solutions
        num = np.arange(1, TotGlib + 1)
        displace = np.column_stack((num, disp))
        print("Nodal displacements:")
        print(displace)

        print("\nElement stresses:")
        for i in range(self.Num_el):
             print(f"{i+1} {stress[i, :]}")

        return displace, stress, strain