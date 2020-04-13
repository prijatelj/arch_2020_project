# Branch Prediction Simulator
- This simulator	implements	various	branch	predictors,	including	simple	1-bit	and	2-bit	predictors,	as	well	as	correlating	(m,n)	predictors
- 	We	will	utilize	two	sets	of	traces	collected	from	a	run	of	gcc.	The	traces	gcc-10K.txt,	and	gcc-8M.txt and contain	~10	thousand	and	~8.5	million	entries respectively
- A	simple	1-bit	predictor	is	(0,1)	predictor	and	simple	2-bit	predictor	is	(0,2)	predictor with	0	global	history.	Therefore,	the simulator	takes as input	a	branch	predictor	type	in	the	form	of	(m,n).

### Executing the script
```sh
$ python branchsim.py -f [filename] -n [n] -m [m] -k [k]
```

 - *m* is global branch history and takes value in the range (0-11)
 - *k* is the number of LSB bits of PC and takes value in the range (1-12)
 - *n* is the branch predictor type and takes value 1 or 2
 - *f* input file name
 - Default values for m,n and k are  6, 1 and 8 respectively

### Example
- (6,1) branch predictor 
```sh
$ python branchsim.py -f gcc-10K.txt -n 1 -m 6 -k 8
```
- 1-bit predictor (0,1)
```sh
$ python branchsim.py -f gcc-10K.txt -n 1 -m 0 -k 8
```
- 2-bit predictor (0,2)
```sh
$ python branchsim.py -f gcc-10K.txt -n 2 -m 0 -k 8
```
