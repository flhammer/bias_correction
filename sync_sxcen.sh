#!/bin/bash

rsync \
	-av \
	--exclude /.git/ \
	--delete \
	~/Projets/bias_correction/ merzisenh@sxcen.cnrm.meteo.fr:~/RSYNCED_CODES/bias_correction/
