test: george/george.h george/george.c george/test.c
	cc -O3 -lamd -lcamd -lccolamd -lcholmod -lcolamd -lcxsparse -lblas -llapack \
		george/george.c george/test.c -o test

clean:
	rm -rf test
