/*********************************************************
	     	     Semana 5 - Programa 12
		Demo - Aprimorando reports com PROC MEANS
**********************************************************
	Sintaxe:
		PROC MEANS DATA=input_table <stat-list>;
			VAR col_name(s);
			CLASS col_name(s);
			WAYS n;
		RUN;

***********************************************************/

ods noproctitle;
/* 1) Configuração padrão */
title "Configuração Padrão";
proc means data=pg1.storm_final;
	var MaxWindMPH;
run;

/* 2) Definindo estatísticas */
title "Adicionando Estatísticas";
proc means data=pg1.storm_final mean median min max maxdec=0;
	var MaxWindMPH;
run;

/* 3) Adicionando variáveis */
title "Classificando por BasinName";
proc means data=pg1.storm_final mean median min max maxdec=0;
	var MaxWindMPH;
	class BasinName;
run;

/* 4) Mais variáveis */
title "Classificando por BasinName e StormType";
proc means data=pg1.storm_final mean median min max maxdec=0;
	var MaxWindMPH;
	class BasinName StormType;
run;

/* 5) Definindo reports com ways */
title "Ways 0, 1 e 2";
proc means data=pg1.storm_final mean median min max maxdec=0;
	var MaxWindMPH;
	class BasinName StormType;
	ways 0 1 2;
	/* Ways define a visualização pela quantidade de variáveis definidas em CLASS */
run;