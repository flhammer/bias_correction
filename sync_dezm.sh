#!/bin/bash

rsync \
	-av \
	--exclude /.git/ \
	--delete \
	~/Projets/bias_correction/ dezm:~/bias_correction/
