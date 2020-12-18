#!/usr/bin/env python
from os import system, getcwd, chdir,listdir


gulpsrc='/home/gfeng/gulp/gulp-5.0/Src'

cwd = getcwd()
chdir(gulpsrc)

# system('cp /media/gfeng/BOND')
system('./mkgulp')
chdir(cwd)
system('/home/feng/gulp/gulp-5.0/Src/gulp<inp-gulp >gulp.out')

# system('grep debug out')
