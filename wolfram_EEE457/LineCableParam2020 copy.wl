(* ::Package:: *)

(* Set of functions to evaluate Z and Y matrices for transmission lines and underground cables *)
(* new version started in 2013-jul-4 it is based on LCparam.m but using only compiled functions
   to speed things up*)

(* Mathematica Raw Program *)
(* first come uncompiled versions *)

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
      
Z2[omega_,rcond_,rins1_,Mur_:1,Mu_:(4*Pi)/10^7]:=((\[ImaginaryJ]*omega*Mur*Mu/(2*\[Pi]))*Log[rins1/rcond])

Z3[Omega_,rins1_,rsheath_,Rhosheath_,Mur_:1,Mu_:(4*Pi)/10^7]:=With[{Etasheath = N[Sqrt[(I*Omega*Mur*Mu)/Rhosheath]]},
With[{Den=BesselI[1,Etasheath*rsheath]*BesselK[1,Etasheath*rins1]-BesselI[1,Etasheath*rins1]*BesselK[1,Etasheath*rsheath]},

((Rhosheath*Etasheath)/(2*\[Pi]*rins1*Den)*(BesselI[0,Etasheath*rins1]*BesselK[1, Etasheath*rsheath]+
BesselK[0,Etasheath*rins1]*BesselI[1,Etasheath*rsheath]))]]

Z4[Omega_,rins1_,rsheath_,Rhosheath_,Mur_,Mu_:(4*Pi)/10^7]:=With[{Etasheath = N[Sqrt[(I*Omega*Mur*Mu)/Rhosheath]]},
With[{Den=BesselI[1,Etasheath*rsheath]*BesselK[1,Etasheath*rins1]-BesselI[1,Etasheath*rins1]*BesselK[1,Etasheath*rsheath]},
(Rhosheath /(2*\[Pi]*rins1*rsheath*Den))]]

Z6[omega_,rsheath_,rins2_,Mur_,Mu_:(4*Pi)/10^7]:=((\[ImaginaryJ]*omega*Mur*Mu/(2*\[Pi]))*Log[rins2/rsheath])

ZSolo[\[Omega]_,r_,h1_,h2_,\[Sigma]solo_,\[Mu]_:4 \[Pi] 10^-7]:=With[{\[Eta]solo=Sqrt[I*\[Omega]*\[Mu]*\[Sigma]solo]},(\[ImaginaryJ]*(\[Omega]*\[Mu])/(2*\[Pi]))*(BesselK[0,\[Eta]solo*Sqrt[r^2+(h1-h2)^2]]+
((h1+h2)^2-r^2)/(r^2+(h1+h2)^2)*(BesselK[2,\[Eta]solo*Sqrt[r^2+(h1+h2)^2]]-
2*(Exp[-(h1+h2)*\[Eta]solo]*(1+(h1+h2)*\[Eta]solo))/((\[Eta]solo^2)*(r^2+(h1+h2)^2))))]

(* Y1[Omega_,rin_,rins1_,epsrins1_,\[Epsilon]_:8.854*10^-12]:=(\[ImaginaryJ]*2*\[Pi]*Omega*(epsrins1*\[Epsilon])/Log[rins1/rin])

Y2[Omega_,rsheath_,rins2_,epsrins2_,\[Epsilon]_:8.854*10^-12]:=(\[ImaginaryJ]*2*\[Pi]*Omega*(epsrins2*\[Epsilon])/Log[rins2/rsheath])

Y3[Omega_,rarm_,rins3_,epsrins3_,\[Epsilon]_:8.854*10^-12]:=(\[ImaginaryJ]*2*\[Pi]*Omega*(epsrins3*\[Epsilon])/Log[rins3/rarm]) *)

Yci[Omega_,rcond_,rins_,\[Epsilon]rins_,\[Epsilon]0_:8.854*10^-12]:=(\[ImaginaryJ]*2*\[Pi]*Omega*(\[Epsilon]rins*\[Epsilon]0)/Log[rins/rcond])




(* rotinas para copiar o linspace e o logspace do matlab *)
logspac = Compile[{{i,_Real},{f,_Real},{np,_Integer}},
                 Table[10.0^(i + (x*(f - i))/(Floor[np] - 1)), {x, 0, np - 1}]
                 ]
     
linspac = Compile[{{i,_Real}, {f,_Real}, {np,_Integer}}, 
                 Table[i + (x*(f - i))/(Floor[np] - 1), {x, 0, np - 1}]
                 ]
        
(*  ground wires are not explicitly considered so we include their effects via Kron Reduction -- matrizes devem estar no formato de impedancia *)        
eliprc = Compile[{{m,_Complex,2},{nc,_Integer},{np,_Integer}},
	            Take[Inverse[m], nc - np, nc - np]
	            ]
	            
(*  reduce bundle to equivalent phase conductors -- matrizes devem estar em formato de admitancia *)
elibndc = Compile[{{aux,_Complex,2},{nb,_Integer},{nf,_Integer}},
	              Table[Sum[aux[[i, j]], {i, nb*m - (nb - 1), nb*m}, {j, nb*n - (nb - 1), nb*n}],
                       {m, nf},{n, nf}]
                 ]


(* internal impedance of cylindrical conductors *)
zintc= Compile[{{omega, _Complex}, {Rhoc, _Real}, {rf, _Real}, {ri,_Real}, {Mur, _Real}},
 ZintTubo[omega, Rhoc, rf, ri, Mur]]
 
zic= Compile[{{omega,_Complex},{rhoc, _Real},{rf, _Real}, {mur,_Real}},
	 Zin[omega, rhoc, rf, mur]]



  
(* external impedance of overhead lines *)
(* with ground wires *)
zextc = Compile[{{omega, _Complex}, {sigma, _Complex}, {npr,_Integer},
	             {x, _Real, 1}, {y, _Real, 1}, {rf, _Real}, {rpr, _Real}}, 
        Module[{Mu = 4.*10^-7*Pi, nc = Length[x], p, Deltaxij, yij, Deltayij},
        p = Sqrt[1.0/(I*omega*Mu*sigma)];
        I*omega*Mu/(2*Pi)*
        Table[
        	If[i != j, 
        	Deltaxij = (x[[i]] - x[[j]]);
        	yij = y[[i]] + y[[j]]; 
        	Deltayij = y[[i]] - y[[j]];
        	0.5*Log[(Deltaxij^2 + (2*p + yij)^2)/(Deltaxij^2 + (Deltayij)^2)],
        	If[i <= nc - npr,
        	Log[2.*(y[[i]] + p)/rf],
        	Log[2.*(y[[i]] + p)/rpr]]], {i, nc}, {j, nc}]]]



(* 
  evaluate impedance and admittannce matrices per unit of length 
  using complex ground plane 
  as overloading in Mma does not work with compiled functions using uncompiled version
 *)  
 (* case 1: with ground wires and bundled conductors  *)
cZYlt[omega_,(x_)?VectorQ, (y_)?VectorQ, sigmas_, rdc_, rf_, rint_, npr_, rdcpr_, rpr_, nb_]:=
    Module[{mu=4*Pi*10^-7, eps=8.854*10^-12, nc=Length[x],  nf, rhoc, rhopr, mp, zin, ze, Z1, Y1},
    	
    nf= IntegerPart[(nc-npr)/nb];
    rhoc= rdc*Pi*(rf^2-rint^2);
    rhopr= rdcpr*Pi*rpr^2;
    
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
         
    Z1 = Inverse[elibndc[eliprc[zin+ze, nc, npr], nb, nf]];     
         
    mp= Table[If[i != j, 
   	    (1/2)*Log[((x[[i]] - x[[j]])^2 + (y[[i]] + y[[j]])^2)/((x[[i]] - x[[j]])^2 + (y[[i]] - y[[j]])^2)], 
        If[i <= nc - npr, Log[(2*y[[i]])/rf], 
        Log[(2*y[[i]])/rpr]]],{i, 1, nc}, {j, 1, nc}];
 
 (* inclusao de condutancia shunt para evitar alguns problemas numericos *)         
    Y1= 3.0*10^-12 DiagonalMatrix[Table[1,{nf}]] +I omega*2*Pi*eps*elibndc[eliprc[mp, nc, npr], nb, nf];
          
    {Z1, Y1}                 

    ];
    
(* case  2: ground wires and unbundled conductors *)
cZYlt[omega_,(x_)?VectorQ, (y_)?VectorQ, sigmas_, rdc_,rf_,rint_, npr_, rdcpr_, rpr_]:=
    Module[{mu=4*Pi*10^-7, eps=8.854*10^-12, nc=Length[x],  nf, rhoc, rhopr, mp, zin, ze, Z1, Y1},
    	
    nf= IntegerPart[(nc-npr)];
    rhoc= rdc*Pi*(rf^2-rint^2);
    rhopr= rdcpr*Pi*rpr^2;
   
   If[ rint != 0, 
   zin= DiagonalMatrix[
   	    Join[zintc[omega, rhoc, rf, rint, 1] Table[1, {i, nc - npr}], 
             zic[omega, rhopr, rpr, 1] Table[1, {i, npr}]]
             ],
   zin= DiagonalMatrix[
   	    Join[zic[omega, rhoc, rf, 1] Table[1, {i, nc - npr}], 
             zic[omega, rhopr, rpr,1] Table[1, {i, npr}]]
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
          
    Y1=3.0*10^-12 DiagonalMatrix[Table[1,{nf}]] + I omega*2*Pi*eps* eliprc[mp, nc, npr];
          
    {Z1, Y1}       
];
     
     
(* case 3: no ground wires or bundled conductors *)   
cZYlt[omega_,(x_)?VectorQ, (y_)?VectorQ, sigmas_, rdc_,rf_,rint_]:=
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
         
    Z1 =  zin+ze ;     
         
    mp= Table[If[i != j, 
   	    (1/2)*Log[((x[[i]] - x[[j]])^2 + (y[[i]] + y[[j]])^2)/((x[[i]] - x[[j]])^2 + (y[[i]] - y[[j]])^2)], 
   	    Log[(2*y[[i]])/rf]], {i, 1, nc}, {j, 1, nc}];
          
    Y1= 3.0*10^-12 DiagonalMatrix[Table[1,{nf}]] + I omega*2*Pi*eps*Inverse[mp]; 
        
    {Z1, Y1}      
];
         
(*caso: configura\[CCedilla]\[ATilde]o flat condutor + isolante*)  
teste=DisplayForm[StyleBox["Voc\[EHat] falhou no teste de sanidade!!! Resumindo, \[EAcute] um insano!!!!!!
Beirando ao mais completo e total desespero!!!!!Inseriu amalucadamente 
valores aleat\[OAcute]rios!!!!!Confira seus valores!!! Auf wiederlesen!!!.","Title"]];
  		
(*cZYsc[omega_,(x__)?VectorQ, (h__)?VectorQ, (r__)?VectorQ,sigmas_,\[Epsilon]ins1__,\[Rho]c__]:=If[aaa== 0,
    Module[{\[Mu]0=4*Pi*10^-7, \[Epsilon]0=8.854*10^-12, z1, z2, z7, zp, zm12, zm13, y1, y2, Z, Y},
    	z1 = Zin[omega, \[Rho]c, r[[1]], 1];
		z2 = Z2[omega,r[[1]],r[[2]],1,\[Mu]0];
		z7 = ZSolo[omega,r[[2]],h[[1]],h[[1]],sigmas,\[Mu]0];

		zp = z1+z2+z7; 
		zm12 = ZSolo[omega,Abs[x[[1]]-x[[3]]],h[[1]],h[[2]],sigmas,\[Mu]0];
		zm13 = ZSolo[omega,Abs[x[[1]]-x[[5]]],h[[1]],h[[3]],sigmas,\[Mu]0];
		
		y1 = Yci[omega,r[[1]],r[[2]],\[Epsilon]ins1,\[Epsilon]0];

		Z={{zp,zm12,zm13},{zm12,zp,zm12},{zm13,zm12,zp}};
		Y=DiagonalMatrix[{y1,y1,y1}];
		
        {Z, Y}       
	    ],teste];*)


(*Cabo blindado*)
cZYsc[omega_,(x_)?VectorQ,(h_)?VectorQ,(r_)?VectorQ,sigmas_,\[Rho]cond_,\[Epsilon]ins1_,\[Rho]blind_,\[Epsilon]ins2_]:=
    Block[{\[Mu]0=4*Pi*10^-7, \[Epsilon]0=8.854*10^-12, z1, z2, z3, z4, z5, z6, z7, zp,zm12, zm13, zsolo1,
		rext,zsolo2,y1, y2,Ycond, Z, Y,k,ncabos,aux,aux2,Zcp,solo,spem},
    	z1 = Zin[omega, \[Rho]cond, r[[1]], 1];
		z2 = Z2[omega,r[[1]],r[[2]],1,\[Mu]0];
		z3 = Z3[omega,r[[2]],r[[3]],\[Rho]blind,1,\[Mu]0];
		z4 = Z4[omega,r[[2]],r[[3]],\[Rho]blind,1,\[Mu]0];
		z5 = ZintTubo[omega, \[Rho]blind, r[[4]], r[[3]]];
		z6 = Z6[omega,r[[3]],r[[4]],1,\[Mu]0];
		 
		zp = {{z1+z2+z3-2*z4+z5+z6,z5+z6-z4},{z5+z6-z4,z5+z6}};
    	ncabos=Length[x];(*n\[UAcute]mero de fases, no caso, o n\[UAcute]mero de cabos*)
		k=2*ncabos;
		aux={{1,1},{1,1}};(*impedancia do solo. Usei a identidade para criar a matrix. As respectivas posi\[CCedilla]\[OTilde]es ser\[ATilde]o substituidas pelos valores corretos*)
		Zcp=Normal[SparseArray[{Band[{1,1},{k,k}]->{zp}},{k,k}]];
		rext=r[[4]];
		spem=Table[zz[i,j]=If[i==j,ZSolo[omega,rext,h[[i]],h[[j]],sigmas,\[Mu]0],
						ZSolo[omega,Abs[x[[i]]-x[[j]]],h[[i]],h[[j]],sigmas,\[Mu]0]],{i,ncabos},{j,ncabos}];
						
		solo=ArrayFlatten@Table[spem[[i,j]]*aux,{i,ncabos},{j,ncabos}];
	   
		Z=Zcp+solo;
	
		y1 = Yci[omega,r[[1]],r[[2]],\[Epsilon]ins1,\[Epsilon]0];
		y2 = Yci[omega,r[[3]],r[[4]],\[Epsilon]ins2,\[Epsilon]0];
		Ycond={{y1, -y1},{-y1, y1+y2}};
		Y=Normal[SparseArray[{Band[{1,1},{k,k}]->{Ycond}},{k,k}]];
        {Z, Y}       
	    ];


(* Line nodal admittance assembly *)

(* Calculo da matriz de admitancia nodal a partir de parametros unitarios e 
   comprimento do circuito *)
ynLT[Z_,Y_,length_]:=
     Module[{Z1=Z, Y1=Y, eval, evect, d, Tv, Tvi, hm, Am, Bm, y11, y12},
            {eval, evect} = Eigensystem[N[Z1.Y1]];
             d = Sqrt[eval];
             Tv = Transpose[evect];
             Tvi = Inverse[Transpose[evect]];
             hm = Exp[-d*length];
             Am = d*(1 + hm^2)/(1 - hm^2);
             Bm = -2.0*d*hm/(1 - hm^2);
             y11 = N[Inverse[Z1]].Tv.DiagonalMatrix[Am].Tvi;
             y12 = N[Inverse[Z1]].Tv.DiagonalMatrix[Bm].Tvi;
            {y11,y12} 
   
];



