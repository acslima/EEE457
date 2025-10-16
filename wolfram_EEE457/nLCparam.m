(* ::Package:: *)

(* Mathematica Raw Program *)
(* nova rotina para o calculo de parametros unitarios em linhas de transmissao e cabos enterrados
   eh uma continuacao do LCparam desenvolvido no Mathematica 5.0 *)

(* arquivo contendo a versao compilada das funcoes para o 
   calculo das matrizes de impedancia e admitancia de 
   linhas de transmissao e cabos subterraneos
   -- na versao atual para cabos sao apenas considerados
   cabos SC (single core )
 *)

(* inicio 2014-09-23  -- atualizado em 2015-10 *)

(* primeiro as funcoes nao compiladas *)
(* versao nao compilada  -- desempenho computacional nao parece mudar se usar compilado aqui ou nao *)
(* a matriz de entrada  no formato de admitancia 
   saida no formato de admitancia tb *)
eliminaFeixe[aux_?MatrixQ, nb_, nf_] := Table[
      	Sum[aux[[i, j]], {i, nb*m - (nb - 1), nb*m},{j, nb*n - (nb - 1), nb*n}],
            {m, nf}, {n, 1, nf}]
            

(* impedancia interna de condutores tubulares *)
ZintTubo[Omega_, Rhoc_, rf_, rint_, Mur_: 1, Mu_: (4.*Pi)/10^7] := 
 With[{Etac = N[Sqrt[(I*Omega*Mur*Mu)/Rhoc]], ri = rint + 10^-6}, 
  With[{Den = 
     BesselK[1, Etac*ri]*BesselI[1, Etac*rf] - 
      BesselK[1, Etac*rf]*BesselI[1, Etac*ri], 
    Num = BesselK[1, Etac*ri]*BesselI[0, Etac*rf] + 
      BesselK[0, Etac*rf]*BesselI[1, Etac*ri]}, ((Rhoc*Etac)*
      Num)/((2*Pi*rf)*Den)]]
      
(* impedancia interna de condutor sem alma de aco *)
(* impedancia interna de condutores cilindricos *) 
(*Mur=90 para cables de aco*)    
Zin[Omega_, Rhopr_, rpr_, Mur_:90, Mu_:(4*Pi)/10^7] := 
    With[{Etapr = Sqrt[(I*Omega*Mu*Mur)/Rhopr]}, 
         ((Etapr*Rhopr)*BesselI[0, Etapr*rpr])/((2*Pi*rpr)*BesselI[1,Etapr*rpr])]
      

(* rotinas para copiar o linspace e o logspace do matlab *)
logspac = Compile[{{i,_Real},{f,_Real},{np,_Integer}},
                 Table[10.0^(i + (x*(f - i))/(Floor[np] - 1)), {x, 0, np - 1}]]
     
linspac = Compile[{{i,_Real}, {f,_Real}, {np,_Integer}}, 
                 Table[i + (x*(f - i))/(Floor[np] - 1), {x, 0, np - 1}]]
        
(*  ground wires are not explicitly considered so we include their effects via Kron Reduction --
 matrize de entrada no formato de impedancia
 saida no formato de admitancia  *)        
eliprc = Compile[{{m,_Complex,2},{nc,_Integer},{np,_Integer}},
	            Take[Inverse[m], nc - np, nc - np], CompilationTarget -> "C", RuntimeAttributes -> Listable]
	            
(*  reduce bundle to equivalent phase conductors -- matrizes devem estar em formato de admitancia *)
(* a matriz de entrada  no formato de admitancia 
   saida no formato de admitancia tb *)
elibnd = Compile[{{aux,_Complex,2},{nb,_Integer},{nf,_Integer}},
	     Table[Sum[aux[[i, j]], {i, nb*m - (nb - 1), nb*m}, {j, nb*n - (nb - 1), nb*n}],{m, nf},{n, nf}],
	     CompilationTarget -> "C", RuntimeAttributes -> Listable]


(* impedancia interna de condutores compiladas *)
zintc= Compile[{{omega, _Complex}, {Rhoc, _Real}, {rf, _Real}, {ri,_Real}, {Mur, _Real}},
       ZintTubo[omega, Rhoc, rf, ri, Mur],
       CompilationTarget -> "C", RuntimeAttributes -> Listable]
 
zic= Compile[{{omega,_Complex},{rhoc, _Real},{rf, _Real}, {mur,_Real}},
	 Zin[omega, rhoc, rf, mur], 
	 CompilationTarget -> "C", RuntimeAttributes -> Listable]


(* 
  montagem das matrizes de impedancia e admitancia por unidade de comprimento para linhas aereas
  solo representado pelo plano complexo 
  as versoes compiladas nao aceitam o polimorfismo por isso vou usar a chamada nao compilada para montar 
  as matrizes e os calculos/reducoes de matrizes da dimensao de condutores fisicos para condutores equivalentes
  eh realizada pela chamada nao compilada
 *)  
 (* alguns resultados mostraram que compilar blocos nao trouxe grandes ganhos
    para a montagem das matrizes passei a usar chamada convencioncal mas empregando
    funcoes compiladas para o calculo *)
 
 (* calculo das matrizes de impedancias
    entrada sao dados geometricos e da rede
    saida sao as impedancias por unidade de comprimento Z1 e Y1  
  *)
    
 (* cenario 1: com pr e com feixe nas fases *)
 calcZYlt[omega_,(x_)?VectorQ, (y_)?VectorQ, sigmas_, rdc_, rf_, rint_, npr_, rdcpr_, rpr_, nb_]:=
    Module[{mu=4*Pi*10^-7, eps=8.854*10^-12, nc=Length[x],  nf, rhoc, rhopr, mp, zin, ze, Z1, Y1},
    	
    nf= IntegerPart[(nc-npr)/nb];
    rhoc= rdc*Pi*(rf^2-rint^2);
    rhopr= rdcpr*Pi*rpr;
    
   zin= DiagonalMatrix[
   	    Join[zintc[omega, rhoc, rf, rint, 1] Table[1, {i, nc - npr}], 
             zic[omega, rhopr, rpr, 90] Table[1, {i, npr}]]
             ];
             
   ze= I omega mu/(2 Pi) With[{p = Sqrt[1/(I*omega*mu*sigmas)]}, 
         Table[If[i != j, 
         	  (Log[((x[[i]] - x[[j]])^2 + (2*p + y[[i]] + y[[j]])^2)/((x[[i]] - x[[j]])^2 +(y[[i]] - y[[j]])^2)])/(2), 
              If[i <= nc - npr, 
               (Log[(2*(y[[i]] + p))/rf]), 
               (Log[(2*(y[[i]] + p))/rpr])]],
         {i, 1, nc}, 
         {j, 1, nc}]];
         
    Z1 = Inverse[elibnd[eliprc[zin+ze, nc, npr], nb, nf]];     
         
    mp= Table[If[i != j, 
   	    (1/2)*Log[((x[[i]] - x[[j]])^2 + (y[[i]] + y[[j]])^2)/((x[[i]] - x[[j]])^2 + (y[[i]] - y[[j]])^2)], 
        If[i <= nc - npr, Log[(2*y[[i]])/rf], 
        Log[(2*y[[i]])/rpr]]],{i, 1, nc}, {j, 1, nc}];
 
 (* inclusao de condutancia shunt para evitar alguns problemas numericos *)         
    Y1= 3.0*10^-12 DiagonalMatrix[Table[1,{nf}]] +I omega*2*Pi*eps*elibnd[eliprc[mp, nc, npr], nb, nf];
          
    {Z1, Y1}                 

    ];
    
(* cenario 2: com pr e sem feixe nas fases *)
calcZYlt[omega_,(x_)?VectorQ, (y_)?VectorQ, sigmas_, rdc_,rf_,rint_, npr_, rdcpr_, rpr_]:=
    Module[{mu=4*Pi*10^-7, eps=8.854*10^-12, nc=Length[x],  nf, rhoc, rhopr, mp, zin, ze, Z1, Y1},
    	
    nf= IntegerPart[(nc-npr)];
    rhoc= rdc*Pi*(rf^2-rint^2);
    rhopr= rdcpr*Pi*rpr;
   
   If[ rint != 0, 
   zin= DiagonalMatrix[
   	    Join[zintc[omega, rhoc, rf, rint, 1] Table[1, {i, nc - npr}], 
             zic[omega, rhopr, rpr, 90] Table[1, {i, npr}]]
             ],
   zin= DiagonalMatrix[
   	    Join[zic[omega, rhoc, rf, 1] Table[1, {i, nc - npr}], 
             zic[omega, rhopr, rpr, 90] Table[1, {i, npr}]]
             ];
   ];
   
             
   ze= I omega mu/(2 Pi) With[{p = Sqrt[1/(I*omega*mu*sigmas)]}, 
         Table[If[i != j, 
         	  (Log[((x[[i]] - x[[j]])^2 + (2*p + y[[i]] + y[[j]])^2)/((x[[i]] - x[[j]])^2 +(y[[i]] - y[[j]])^2)])/(2), 
              If[i <= nc - npr, 
               (Log[(2*(y[[i]] + p))/rf]), 
               (Log[(2*(y[[i]] + p))/rpr])]],
         {i, 1, nc}, 
         {j, 1, nc}]];
         
    Z1 = Inverse[eliprc[zin+ze, nc, npr]];     
         
    mp= Table[If[i != j, 
   	    (1/2)*Log[((x[[i]] - x[[j]])^2 + (y[[i]] + y[[j]])^2)/((x[[i]] - x[[j]])^2 + (y[[i]] - y[[j]])^2)], 
        If[i <= nc - npr, Log[(2*y[[i]])/rf], 
        Log[(2*y[[i]])/rpr]]],{i, 1, nc}, {j, 1, nc}];
          
    Y1=3.0*10^-12 DiagonalMatrix[Table[1,{nf}]] + I omega*2*Pi*eps*eliprc[mp, nc, npr];
          
    {Z1, Y1}       
];
     
     
  (* cenario 3: sem pr e sem feixe nas fases *)   
    calcZYlt[omega_,(x_)?VectorQ, (y_)?VectorQ, sigmas_, rdc_,rf_,rint_]:=
    Module[{mu=4*Pi*10^-7, eps=8.854*10^-12, nc=Length[x],  nf, rhoc, mp, zin, ze, Z1, Y1},
    	
    nf= (nc);
    rhoc= rdc*Pi*(rf^2-rint^2);
    
   
   If[ rint != 0, 
   zin= DiagonalMatrix[zintc[omega, rhoc, rf, rint, 1] Table[1, {i, nc}]],
   zin= DiagonalMatrix[zic[omega, rhoc, rf, 1] Table[1, {i, nc}]]
   ];
   
             
   ze= I omega mu/(2 Pi) With[{p = Sqrt[1/(I*omega*mu*sigmas)]}, 
         Table[If[i != j, 
         	  (Log[((x[[i]] - x[[j]])^2 + (2*p + y[[i]] + y[[j]])^2)/((x[[i]] - x[[j]])^2 +(y[[i]] - y[[j]])^2)])/(2), 
         	  (Log[(2*(y[[i]] + p))/rf])],
         {i, 1, nc}, 
         {j, 1, nc}]];
         
    Z1=  zin+ze ;     
         
    mp= Table[If[i != j, 
   	    (1/2)*Log[((x[[i]] - x[[j]])^2 + (y[[i]] + y[[j]])^2)/((x[[i]] - x[[j]])^2 + (y[[i]] - y[[j]])^2)], 
   	    Log[(2*y[[i]])/rf]], {i, 1, nc}, {j, 1, nc}];
          
    Y1= 3.0*10^-12 DiagonalMatrix[Table[1,{nf}]] + I omega*2*Pi*eps*Inverse[mp]; 
        
    {Z1, Y1}      
];
     
     
(* funcoes adicionais para a montagem da matriz de admitancia nodal e manipulacao de quadripolos *)

(* 
  Calculo da matriz de admitancia nodal a partir de parametros unitarios e 
  comprimento do circuito 
  nao adianta tentar compilar pois o Eigensystems nao aceita ser compilado 
*)
Ynodal[Z_,Y_,length_]:=
     Module[{Z1=Z,Y1=Y,eval,evect,d,Tv,Tvi,hm,Am,Bm,y11,y12},
     	
            {eval, evect} = Eigensystem[N[Z1.Y1]];
             d = Sqrt[eval];
             Tv = Transpose[evect];
             Tvi = Inverse[Transpose[evect]];
        
             hm = Exp[-d*length];
             Am = d*(1 + hm^2)/(1 - hm^2);
             Bm = -2.0*d*hm/(1 - hm^2);
        
             y11 = Inverse[Z1].Tv.DiagonalMatrix[Am].Tvi;
             y12 = Inverse[Z1].Tv.DiagonalMatrix[Bm].Tvi;
             
             {y11,y12} 
     ]

 
	            
(* funcao para conversao de Ynodal simetrica para quadripolo *)
Y2Quad[y11_, y12_]:=
     Module[{aux,AA,BB,CC},
             aux = Inverse[y12];
             AA = -aux.y11;
             BB = -aux;
             CC = y12 - y11.aux.y11;
             ArrayFlatten[{{AA, BB}, {CC, AA}}]
  ]                   

 (* funcao para conversao do quadripolo para Ynodal *)
 (* usa apenas os autovalores menores para evitar overflow numerico *)    
Quad2Ym[mat_]:=
  Module[{n,d,Tv,Tvi,Aeq,Beq,Ceq,Deq,aux2,yss,ysr,yrs,yrr,eval,evect},
  	
  	n = Round[Length[mat]/2];
  	Aeq = mat[[Range[n], Range[n] ]];
  	Beq = mat[[Range[n], Range[n+1,2*n] ]];
  	Ceq = mat[[Range[n+1,2*n], Range[n] ]];
  	Deq = mat[[Range[n+1,2*n], Range[n+1,2*n] ]];
  	
  	{eval, evect} = Eigensystem[N[Beq]];
  	d = 1/eval;
    Tv = Transpose[evect];
    Tvi = Inverse[Transpose[evect]];
  	aux2 = Tv.DiagonalMatrix[d].Tvi;
  	
  	yss = Deq.aux2;
  	ysr = Ceq - Deq.aux2. Aeq;
  	yrs = -aux2;
    yrr = aux2.Aeq;
    
    ArrayFlatten[{{yss,ysr},{yrs,yrr}}] 
  ]
  
  
  
