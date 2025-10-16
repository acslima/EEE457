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
Zin[Omega_, Rhopr_, rpr_, Mur_:1, Mu_:(4*Pi)/10^7] := 
    With[{Etapr = Sqrt[(I*Omega*Mu*Mur)/Rhopr]}, 
         ((Etapr*Rhopr)*BesselI[0, Etapr*rpr])/((2*Pi*rpr)*BesselI[1,Etapr*rpr])]
      
Z2[omega_,rcond_,rins1_,Mur_:1,Mu_:(4*Pi)/10^7]:=((I*omega*Mur*Mu/(2*\[Pi]))*Log[rins1/rcond])

Z3[Omega_,rins1_,rsheath_,Rhosheath_,Mur_:1,Mu_:(4*Pi)/10^7]:=With[{Etasheath = N[Sqrt[(I*Omega*Mur*Mu)/Rhosheath]]},
With[{Den=BesselI[1,Etasheath*rsheath]*BesselK[1,Etasheath*rins1]-BesselI[1,Etasheath*rins1]*BesselK[1,Etasheath*rsheath]},

((Rhosheath*Etasheath)/(2*\[Pi]*rins1*Den)*(BesselI[0,Etasheath*rins1]*BesselK[1, Etasheath*rsheath]+
BesselK[0,Etasheath*rins1]*BesselI[1,Etasheath*rsheath]))]]

Z4[Omega_,rins1_,rsheath_,Rhosheath_,Mur_:1,Mu_:(4*Pi)/10^7]:=With[{Etasheath = N[Sqrt[(I*Omega*Mur*Mu)/Rhosheath]]},
With[{Den=BesselI[1,Etasheath*rsheath]*BesselK[1,Etasheath*rins1]-BesselI[1,Etasheath*rins1]*BesselK[1,Etasheath*rsheath]},
(Rhosheath /(2*\[Pi]*rins1*rsheath*Den))]]

Z6[omega_,rsheath_,rins2_,Mur_:1,Mu_:(4*Pi)/10^7]:=((I*omega*Mur*Mu/(2*\[Pi]))*Log[rins2/rsheath])

ZSolo[\[Omega]_,r_,h1_,h2_,\[Sigma]solo_,\[Mu]_:4 \[Pi] 10^-7]:=With[{\[Eta]solo=Sqrt[I*\[Omega]*\[Mu]*\[Sigma]solo]},(I*(\[Omega]*\[Mu])/(2*\[Pi]))*(BesselK[0,\[Eta]solo*Sqrt[r^2+(h1-h2)^2]]+
((h1+h2)^2-r^2)/(r^2+(h1+h2)^2)*(BesselK[2,\[Eta]solo*Sqrt[r^2+(h1+h2)^2]]-
2*(Exp[-(h1+h2)*\[Eta]solo]*(1+(h1+h2)*\[Eta]solo))/((\[Eta]solo^2)*(r^2+(h1+h2)^2))))]

(* Y1[Omega_,rin_,rins1_,epsrins1_,\[Epsilon]_:8.854*10^-12]:=(\[ImaginaryJ]*2*\[Pi]*Omega*(epsrins1*\[Epsilon])/Log[rins1/rin])

Y2[Omega_,rsheath_,rins2_,epsrins2_,\[Epsilon]_:8.854*10^-12]:=(\[ImaginaryJ]*2*\[Pi]*Omega*(epsrins2*\[Epsilon])/Log[rins2/rsheath])

Y3[Omega_,rarm_,rins3_,epsrins3_,\[Epsilon]_:8.854*10^-12]:=(\[ImaginaryJ]*2*\[Pi]*Omega*(epsrins3*\[Epsilon])/Log[rins3/rarm]) *)

Yci[Omega_,rcond_,rins_,\[Epsilon]rins_,\[Epsilon]0_:8.854*10^-12]:=(I*2*\[Pi]*Omega*(\[Epsilon]rins*\[Epsilon]0)/Log[rins/rcond])




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
 ZintTubo[omega, Rhoc, rf, ri, Mur]];
 
zic= Compile[{{omega,_Complex},{rhoc, _Real},{rf, _Real}, {mur,_Real}},
	 Zin[omega, rhoc, rf, mur]];



zintc= Compile[{{omega, _Complex}, {Rhoc, _Real}, {rf, _Real}, {ri,_Real}, {Mur, _Real}},
 ZintTubo[omega, Rhoc, rf, ri, Mur]];
 
zic= Compile[{{omega,_Complex},{rhoc, _Real},{rf, _Real}, {mur,_Real}},
	 Zin[omega, rhoc, rf, mur]];


  
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
             ];];
   
             
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
          
    {Z1, Y1}];
     
     
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
        
    {Z1, Y1}];
         
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
cZYsc[omega_,(h_)?VectorQ,(r_)?VectorQ,\[Rho]c_,\[Epsilon]1_,\[Rho]b_,\[Epsilon]2_]:=
    Module[{\[Mu]0=4*Pi*10^-7, \[Epsilon]0=8.854*10^-12, z1, z2, z3, z4, z5, z6, Zp, y1, y2,Yp},
    	z1 = Zin[omega, \[Rho]c, r[[1]], 1];
		z2 = Z2[omega,r[[1]],r[[2]],1,\[Mu]0];
		z3 = Z3[omega,r[[2]],r[[3]],\[Rho]b,1,\[Mu]0];
		z4 = Z4[omega,r[[2]],r[[3]],\[Rho]b,1,\[Mu]0];
		z5 = ZintTubo[omega, \[Rho]b, r[[3]], r[[2]], 1, \[Mu]0];
		z6 = Z6[omega,r[[3]],r[[4]],1,\[Mu]0];
		Zp = {{z1+z2+z3-2*z4+z5+z6,z5+z6-z4},{z5+z6-z4,z5+z6}};
		y1 = Yci[omega,r[[1]],r[[2]],\[Epsilon]1,\[Epsilon]0];
		y2 = Yci[omega,r[[3]],r[[4]],\[Epsilon]2,\[Epsilon]0];
		Yp={{y1, -y1},{-y1, y1+y2}};
		{Zp, Yp}];


(*COM ARMADURA*)
cZYsc[omega_,(h_)?VectorQ,(r_)?VectorQ,\[Rho]c_,\[Epsilon]1_,\[Rho]b_,\[Epsilon]2_,\[Rho]a_,\[Epsilon]3_]:=
    Module[{\[Mu]0=4*Pi*10^-7, \[Epsilon]0=8.854*10^-12, z1, z2, z3, z4, z5, z6, z7,z8, z9,z10, z11, zc, zb, za, zcb, zca, zba, zm12, zm13,
		zcabo, zsolo1,zsolo2,y1, y2, y3, Ycond, Z, Y,ncabos,k,aux,Zcp,solo,aux2,rext,Zp,Yp},
    	z1 = Zin[omega, \[Rho]c, r[[1]], 1];
		z2 = Z2[omega,r[[1]],r[[2]],1,\[Mu]0];
		z3 = Z3[omega,r[[2]],r[[3]],\[Rho]b,1,\[Mu]0];
		z4 = Z4[omega,r[[2]],r[[3]],\[Rho]b,1,\[Mu]0];
		z5 = ZintTubo[omega, \[Rho]b, r[[3]], r[[2]], 1, \[Mu]0];
		z6 = Z6[omega,r[[3]],r[[4]],1,\[Mu]0];
        z7 = Z3[omega,r[[4]],r[[5]],\[Rho]a,90,\[Mu]0];
        z8 = Z4[omega,r[[4]],r[[5]],\[Rho]a,90,\[Mu]0];
		z9 = ZintTubo[omega, \[Rho]a, r[[5]], r[[4]],90, \[Mu]0];
		z10 = Z6[omega,r[[5]],r[[6]],1,\[Mu]0];
				
		zc = z1+z2+z3+z5-2*z4+z6+z7+z9-2*z8+z10;
        zb = z5+z6+z7+z9-2*z8+z10;
        za = z9+z10;
        zcb = z5-z4+z6+z7+z9-2*z8+z10;
        zca = z9-z8+z10;
        zba = z9-z8+z10;

        Zp = {{zc,zcb,zca},{zcb,zb,zba},{zca,zba,za}};
    	
		y1 = Yci[omega,r[[1]],r[[2]],\[Epsilon]1,\[Epsilon]0];
		y2 = Yci[omega,r[[3]],r[[4]],\[Epsilon]2,\[Epsilon]0];
		y3 = Yci[omega,r[[5]],r[[6]],\[Epsilon]3,\[Epsilon]0];
		Yp={{y1, -y1, 0},{-y1, y1+y2, -y2},{0, -y2, y2+y3}};
		
        {Zp, Yp}];


cZYsc2[omega_,(h_)?VectorQ,nCond_?NumberQ,(r_)?VectorQ,\[Sigma]s_,\[Rho]c_,\[Epsilon]1_,\[Rho]b_,\[Epsilon]2_(*,\[Rho]a_,\[Epsilon]3_*)]:=
    Module[{\[Mu]0=4*Pi*10^-7, \[Epsilon]0=8.854*10^-12, z1, z2, z3, z4, z5, z6, z7,z8, z9,z10, z11, zc, zb, za, zcb, zca, zba, zm12, zm13,
		zcabo, zsolo1,zsolo2,y1, y2, y3, Ycond, Z, Y,ncabos,k,aux,Zcp,solo,aux2,rext,Zp,Yp, 
		\[Mu]rc,\[Mu]rb,\[Mu]ra, 
		withArmour, withSheath
		},

		\[Mu]rc = 1; (* core *)
		\[Mu]rb = 1; (* sheath *)
		\[Mu]ra = 90; (* armour *)
		
		(* check for tubular core *)
		If[r[[1]]<1*10^-12,
			(* solid core *)
			z1 = Zin[omega, \[Rho]c, r[[2]], \[Mu]rc]; (* core self *)	
			,
			(* tubular *)
			z1 = ZintTubo[omega, \[Rho]c, r[[1]],r[[2]], \[Mu]rc]; (* core self *)	
		]

(* Zin[Omega_, Rhopr_, rpr_, Mur_:1, Mu_:(4*Pi)/10^7] *)
(* ZintTubo[Omega_, Rhoc_, rf_, rint_, Mur_: 1, Mu_: (4.*Pi)/10^7] *)    	
		If[nCond>=2,
		  z2 = Z2[omega,r[[2]],r[[3]]];
		  z3 = Z3[omega,r[[3]],r[[4]],\[Rho]b];
		  z4 = Z4[omega,r[[3]],r[[4]],\[Rho]b];
		  z5 = ZintTubo[omega, \[Rho]b, r[[4]], r[[3]], \[Mu]rb]; (* sheath self *)
		  z6 = Z6[omega,r[[4]],r[[5]]];

		  withSheath = 1;
		  ,
          withSheath = 0;
		];
		If[nCond>=3 (* armour *),
          z7 = Z3[omega,r[[5]],r[[6]],\[Rho]a, \[Mu]ra];
          z8 = Z4[omega,r[[5]],r[[6]],\[Rho]a, \[Mu]ra];
		  z9 = ZintTubo[omega, \[Rho]a, r[[6]], r[[5]], \[Mu]ra]; (* armour self *)
		  z10 = Z6[omega,r[[6]],r[[7]]]; (* armor-insulation layer due to field var. *)
		  
		  withArmour = 1;
		  ,
		  withArmour = 0;
		];
		
		(* flags *)
		zc = z1+z2 + withSheath(z3+z5-2*z4+z6) + withArmour(z7+z9-2*z8+z10);
        zb = withSheath(z5+z6) + withArmour(z7-2*z8+z9+z10);
        za = withArmour(z9+z10);
        zcb = withSheath(z5-z4+z6) + withArmour(+z7+z9-2*z8+z10);
        zca = withArmour(z9-z8+z10);
        zba = withArmour(z9-z8+z10);
    	

		y1 = Yci[omega,r[[2]],r[[3]],\[Epsilon]1,\[Epsilon]0];
(* cases: 
core only: 1
core + sheath: 2
core + sheath + armour: 3
*)
Switch[
nCond,
1 (* core only *),
	Zp = zc;
	Yp = y1;
,
2 (* core + sheath *),
	Zp = {{zc, zcb},{zcb, zb}};
	y2 = Yci[omega,r[[4]],r[[5]],\[Epsilon]2,\[Epsilon]0];
	Yp={{y1, -y1},{-y1, y1+y2}};
,
3 (* core + sheath + armour *),	
	Zp = {{zc,zcb,zca},{zcb,zb,zba},{zca,zba,za}};

	y2 = Yci[omega,r[[4]],r[[5]],\[Epsilon]2,\[Epsilon]0];
	y3 = Yci[omega,r[[6]],r[[7]],\[Epsilon]3,\[Epsilon]0];
	Yp={{y1, -y1, 0},{-y1, y1+y2, -y2},{0, -y2, y2+y3}};

];
		
        {Zp, Yp}];


montaZY[omega_,(h_)?VectorQ,(r_)?VectorQ,(x_)?VectorQ,sigmas_,Zp_,Yp_]:=
    Module[{\[Mu]0=4*Pi*10^-7, \[Epsilon]0=8.854*10^-12,dim,Z,Y,Zcp,solo,k,ncabos,spem,aux,rext,zz},
		dim=Length[r];
		ncabos=Length[x];(*n\[UAcute]mero de fases, no caso, o n\[UAcute]mero de cabos*)
		k=Floor[dim/2]*ncabos;
		aux=Normal[SparseArray[{{_,_}->1},{Floor[dim/2],Floor[dim/2]}]];(*impedancia do solo. Usei a identidade para criar a matrix. As respectivas posi\[CCedilla]\[OTilde]es ser\[ATilde]o substituidas pelos valores corretos*)
		Zcp=Normal[SparseArray[{Band[{1,1},{k,k}]->{Zp}},{k,k}]];
		rext=r[[dim]];
		spem=Table[zz[i,j]=If[i==j,
                   ZSolo[omega,rext,h[[i]],h[[j]],sigmas,\[Mu]0],
			       ZSolo[omega,Abs[x[[i]]-x[[j]]],h[[i]],h[[j]],sigmas,\[Mu]0]],{i,ncabos},{j,ncabos}];
		
        solo=ArrayFlatten@Table[spem[[i,j]]*aux,{i,ncabos},{j,ncabos}];
    	
        Z=Zcp+solo;

		Y=Normal[SparseArray[{Band[{1,1},{k,k}]->{Yp}},{k,k}]];

		{Z, Y}];


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
            {y11,y12}];





(* Elimina o Cruzamento de Autovetores *)
(* Metodo de Newton-Raphson *)
(* Elimina o Cruzamento de Autovetores *)
(* Metodo de Newton-Raphson \
*)
eigvNR[G_?MatrixQ, V0_, d0_, Tol_: 10^-12] := 
  Module[{A = G, nc, scale, v, d, vec, res, jac, i, itr, outd, outv},
   nc = Length[A];
   scale = Norm[A];
   A = A/scale;
   outd = ConstantArray[0., nc];
   outv = ConstantArray[0., {nc, nc}];
   itr = 0;
   jac = ConstantArray[0., {nc + 1, nc + 1}];
   Do[{v = V0[[1 ;;, i]];
     d = d0[[i]];
     res = Max[Abs[A.v - v*d]];
     While[(res > Tol) && (itr < 50000),
      itr = itr + 1;
      jac[[1 ;; nc, 1 ;; nc]] = A - d*IdentityMatrix[nc];
      jac[[1 ;; nc, nc + 1]] = -v;
      jac[[nc + 1, 1 ;; nc]] = 2*v;
      res = Join[A.v - v*d, {v.v - 1}];
      vec = LinearSolve[jac, res, Method->"Multifrontal"];
      res = Max[Abs[vec]];
      v = v - vec[[1 ;; nc]];
      d = d - vec[[nc + 1]];
      ];
     outd[[i]] = d*scale;
     outv[[1 ;;, i]] = Normalize[v];}, {i, 1, nc}];
   {outd, outv}
   ]
   
   (* Rotacao da Matriz de Autovetores para minimizar parte Imaginaria *)
(* Complemento ao Metodo de Newton-Raphson *)
rot[s_?MatrixQ]:=Module[{S=s,Nc,SA,SB,scale,ang,scale1,col,scale2,err1,err2,numerator,denominator,aaa,bbb,ccc,ddd,eee,fff,j},
Nc=Length[S];
scale1=scale2=scale=ang=err1=err2=numerator=denominator=ConstantArray[0.,Nc];
SA=SB=ConstantArray[0.,{Nc,Nc}];

Table[

Table[numerator[[col]]=numerator[[col]]+Im[S[[j,col]]]*Re[S[[j,col]]];
denominator[[col]]=denominator[[col]]+Re[S[[j,col]]]^2-Im[S[[j,col]]]^2
,{j,Nc}];

numerator[[col]]=-2*numerator[[col]];
ang[[col]]=0.5*ArcTan[denominator[[col]],numerator[[col]]];
scale1[[col]]=Cos[ang[[col]]]+I*Sin[ang[[col]]];
scale2[[col]]=Cos[ang[[col]]+\[Pi]/2.]+I*Sin[ang[[col]]+\[Pi]/2.];
aaa=bbb=ccc=ddd=eee=fff=0.;

Table[SA[[j,col]]=S[[j,col]]*scale1[[col]];SB[[j,col]]=S[[j,col]]*scale2[[col]];
aaa=aaa+Im[SA[[j,col]]]^2;bbb=bbb+Re[SA[[j,col]]]*Im[SA[[j,col]]];ccc=ccc+Re[SA[[j,col]]]^2;
ddd=ddd+Im[SB[[j,col]]]^2;eee=eee+Re[SB[[j,col]]]*Im[SB[[j,col]]];fff=fff+Re[SB[[j,col]]]^2
,{j,Nc}];

err1[[col]]=aaa*Cos[ang[[col]]]^2+bbb*Sin[N[2*ang[[col]],16]]+ccc*Sin[ang[[col]]]^2;
err2[[col]]=ddd*Cos[ang[[col]]]^2+eee*Sin[N[2*ang[[col]],16]]+fff*Sin[ang[[col]]]^2;
If[err1[[col]]<err2[[col]],scale[[col]]=scale1[[col]],scale[[col]]=scale2[[col]]];
S[[All,col]]=S[[All,col]]*scale[[col]]

,{col,Nc}];

Return[S]
]
   
