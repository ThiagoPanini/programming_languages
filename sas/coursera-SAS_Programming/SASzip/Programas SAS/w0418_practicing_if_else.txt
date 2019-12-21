/*********************************************************
	     	     Semana 4 - Programa 18
	Prática Level 1 - Criando nova coluna condicional
**********************************************************
1) Submeter o programa inicial e visualizar os resultados

2) Adicionar uma condição para criar uma nova coluna, de
nome ParkType, baseada os tópicos a seguir:

Tipo ; ParkType
NM ; Monument
NP ; Park
NPRE, PRE or PRESERVE ; Preserve
NS ; Seashore
RVR or RIVERWAYS ; River

3) Modificar o comando PROC FREQ pra mostrar um report de
frequência para a coluna ParkType

	a) Qual a frequência da coluna Seashore?

***********************************************************/

proc print data=pg1.np_summary (obs=10);
run;

data summary;
	set pg1.np_summary;
	keep Type ParkType;
	if Type='NM' then do;
		ParkType='Monument';
	end;
	else if Type='NP' then do;
		ParkType='Park';
	end;
	else if Type in ('NPRE', 'PRE', 'PRESERVE') then do;
		Parktype='Preserve';
	end;
	else if Type='NS' then do;
		ParkType='Seashore';
	end;
	else if Type in ('RVR', 'RIVERWAYS') then do;
		ParkType='River';
	end;
	else do 
		ParkType='Not Informed';
	end;
run;

* Ou melhor...;
data park_type;
    set pg1.np_summary;
    length ParkType $ 8;
    if Type='NM' then ParkType='Monument';
    else if Type='NP' then ParkType='Park';
    else if Type in ('NPRE', 'PRE', 'PRESERVE') then
        ParkType='Preserve';
    else if Type in ('RVR', 'RIVERWAYS') then ParkType='River';
    else if Type='NS' then ParkType='Seashore';
run;


proc freq data=park_type;
	tables ParkType;
run;