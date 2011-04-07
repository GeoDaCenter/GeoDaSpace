    """from old weights.py: the kernel weights and kernel lag functions
        check pysal
        http://code.google.com/p/pysal/source/browse/trunk/pysal/weights/Distance.py#105
        http://code.google.com/p/pysal/source/browse/trunk/pysal/weights/tests/test_Distance.py
    """

    def ktriangle(self,rowstand,power,dmax):
        """ktriangle: triangular/Bartlett kernel function
        
           if dmax not specified (default) then variable bandwidth
           dmax is bandwidth
           uses existing distance weights, read in as raw
           no checks for 0.0, fastest but zero weights
           note: is destructive replaces nbwt
        """
        tt0 = time.time()
        if dmax:   # fixed bandwidth
            for i in range(self.np):
                if self.nb[i]:  # no islands
                    self.nbwt[i] = [ 1.0 - j/dmax for j in self.nbwt[i] ]
        else:      # variable bandwidth
            for i in range(self.np):
                if self.nb[i]:
                    dmax = max(self.nbwt[i])
                    self.nbwt[i] = [ 1.0 - j/dmax for j in self.nbwt[i] ]
        tt1 = time.time()
        print "ktriangle",tt1-tt0
        

    def kepanech(self,rowstand,power,dmax):
        """kepanech: Epanechnikov kernel function
        
           if dmax not specified (default) then variable bandwidth
           dmax is bandwidth
           uses existing distance weights, read in as raw
           no checks for 0.0, fastest but zero weights
           rowstand, power are ignored
           note: is destructive replaces nbwt
        """
        tt0 = time.time()
        if dmax:   # fixed bandwidth
            for i in range(self.np):
                if self.nb[i]:  # no islands
                    self.nbwt[i] = [ 1.0 - (j/dmax)**2 for j in self.nbwt[i] ]
        else:      # variable bandwidth
            for i in range(self.np):
                if self.nb[i]:
                    dmax = max(self.nbwt[i])
                    self.nbwt[i] = [ 1.0 - (j/dmax)**2 for j in self.nbwt[i] ]
        tt1 = time.time()
        print "kepanech",tt1-tt0

    def kbisquare(self,rowstand,power,dmax):
        """kbisquare: bisquare kernel function
        
           if dmax not specified (default) then variable bandwidth
           dmax is bandwidth
           uses existing distance weights, read in as raw
           no checks for 0.0, fastest but zero weights
           rowstand,power are ignored
           note: is destructive replaces nbwt
        """
        tt0 = time.time()
        if dmax:   # fixed bandwidth
            for i in range(self.np):
                if self.nb[i]:  # no islands
                    self.nbwt[i] = [ (1.0 - (j/dmax)**2)**2 
                        for j in self.nbwt[i] ]
        else:      # variable bandwidth
            for i in range(self.np):
                if self.nb[i]:
                    dmax = max(self.nbwt[i])
                    self.nbwt[i] = [ (1.0 - (j/dmax)**2)**2 
                        for j in self.nbwt[i] ]
        tt1 = time.time()
        print "kbisquare",tt1-tt0

    def ktriangle2(self,dmax=0):
        """ktriangle2: triangular/Bartlett kernel function
        
           if dmax not specified (default) then variable bandwidth
           dmax is bandwidth
           uses existing distance weights, read in as raw
           note: assumes distances are sorted
           note: is destructive replaces nbwt
        """
        tt0 = time.time()
        if dmax:   # fixed bandwidth
            for i in range(self.np):
                if self.nb[i]:  # no islands
                    self.nbwt[i] = [ 1.0 - j/dmax for j in self.nbwt[i] ]
        else:      # variable bandwidth
            for i in range(self.np):
                if self.nb[i]:
                    dmax = self.nbwt[i][-1]
                    self.nbid[i] = [ j for j in self.nbid[i][:-1] ]
                    self.nbwt[i] = [ 1.0 - j/dmax for j in self.nbwt[i][:-1] ] 
                    self.nb[i] = self.nb[i] - 1
        tt1 = time.time()
        print "ktriangle2",tt1-tt0


    def ksplagl(self,X):
        """splagl: kernel lag operator using lists
        
           note: weight i,i always = 1 even for islands
        """
#        tt0 = time.time()
        if type(X) == list:    # only for lists
            wx = []          # initialize lags
            # islands are dealt with in data structure
            wx = [[ sum([X[i][j]*h for (j,h) in 
                    zip(self.nbid[hh],self.nbwt[hh]) ]) + X[i][hh] for
                    hh in range(self.np)] for i in range(len(X))]
        else:
            return 0
#        tt1 = time.time()
#        print "ksplagl",tt1-tt0
        return wx
        
        """from old spreg hac variance matrices
        
        """
        
                    
    def hacols(self,hacflag=0):
        """hacols: HAC variance estimation for OLS (Kelejian-Prucha)
        
           hacflag: 1 for HAC, 0 for White       
           returns HAC variance
        """        # initialize X as array
        kx = nm.array(self.X)            # N by K array of X
        # initialize residuals as array
        ke = nm.array(self.results['e'])   # N by 1 array of OLS residuals
        # kernel cross-products
        kxe = kx * ke   # this is not matrix multiplication but each X is multiplied by e, so x(ik)*e(i)
        if hacflag:
            wkxe = self.kw1.ksplagl(kxe.tolist())     # this applies KW to kx*ke so to the product of xik.ei
            wkxe = nm.array(wkxe)  # side effect of moving between lists and arrays
            kphi = nm.inner(kxe,wkxe)  # the cross product of the Xe times the Wxe or Sumi Sumj xi xj k(ij) ei ej
            # eq 8 on p 141 of KP
        else:
            kphi = nm.inner(kxe,kxe)  # just the cross product of x*e for White
        # variance matrix as (X'X)-1 Phi (X'X)-1
        kdd = nm.dot(kphi,self.results['xd'])   # xd is (X'X)-1 so this is Phi*(X'X)-1 
        phi = nm.dot(self.results['xd'],kdd) # this is (X'X)-1 (XeKXe) (X'X)-1 the variance matrix for OLS case
        return phi
        
        
    def hac2sls(self,hacflag=0):
        """hacols: HAC variance estimation for 2SLS (Kelejian-Prucha)
        
           hacflag: 1 for HAC, 0 for White       
           returns HAC variance
        """
        # initialize X as array
        kx = nm.array(self.H) # N by P array of instruments H
        # initialize residuals as array
        ke = nm.array(self.results['e']) # N by 1 array of 2SLS residuals
        # kernel cross-products
        kxe = kx * ke   # note kx here is H so this is h(ik)*e(i)
        #print 'THIS IS KXE'
        #print kxe
        if hacflag:
            wkxe = self.kw1.ksplagl(kxe.tolist())  # kernel lag applied to the He
            #print 'this is wkxe'
            #print wkxe
            wkxe = nm.array(wkxe)  # need to turn into an array from a list -- not crucial
            kphi = nm.inner(kxe,wkxe)  # cross product Sumi Sum hir hjs ui uj K(ij)  equation 8 p. 141 in KP
        else:
            kphi = nm.inner(kxe,kxe)  # just the white thing -- we may not need
        # variance matrix as (Zh'Zh)-1Z'H(H'H)-1 Phi (H'H)-1 H'Z(Zh'Zh)-1
        # see theorem 3
        kdd = nm.dot(self.results['xdp'],kphi)  # xdp is  [Z'H (H'H)-1 H'Z]-1 Z'H(H'H)-1 kphi is Phi
        print 'THIS IS KDD'
        print kdd
        phi = nm.dot(kdd,nm.transpose(self.results['xdp']))  #  [Z'H (H'H)-1 H'Z]-1 Z'H(H'H)-1 kphi (H'H)-1 H'Z [Z'H (H'H)-1 H'Z]-1 
        # see proof of theorem 3 p. 151 Zhat'Zhat is replaced by explicit expression
        return phi
        
        