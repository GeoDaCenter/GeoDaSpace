"""
Distribution functions and probabilities module 
----------------------------------------------------------------------
AUTHOR(S):  Sergio J. Rey sjrey@users.sourceforge.net
----------------------------------------------------------------------
Copyright (c) 2000-2005 Sergio J. Rey
======================================================================
This source code is licensed under the GNU General Public License, 
Version 2.  See the file COPYING for more details.
======================================================================
Adapted from original stats.py by
Copyright (c) 1999-2000 Gary Strangman; All Rights Reserved.

This software is distributable under the terms of the GNU
General Public License (GPL) v2, the text of which can be found at
http://www.gnu.org/copyleft/gpl.html. Installing, importing or otherwise
using this module constitutes acceptance of the terms of this License.

Disclaimer

This software is provided "as-is".  There are no expressed or implied
warranties of any kind, including, but not limited to, the warranties
of merchantability and fittness for a given application.  In no event
shall Gary Strangman be liable for any direct, indirect, incidental,
special, exemplary or consequential damages (including, but not limited
to, loss of use, data or profits, or business interruption) however
caused and on any theory of liability, whether in contract, strict
liability or tort (including negligence or otherwise) arising in any way
out of the use of this software, even if advised of the possibility of
such damage.

Comments and/or additions are welcome (send e-mail to:
strang@nmr.mgh.harvard.edu).
 

OVERVIEW

"""


import math


def chicdf(chisq,df):
    """
Returns the (1-tailed) probability value associated with the provided
chi-square value and df.  Adapted from chisq.c in Gary Perlman's |Stat.

Usage:   chicdf(chisq,df)
"""
    BIG = 20.0
    def ex(x):
	BIG = 20.0
	if x < -BIG:
	    return 0.0
	else:
	    return math.exp(x)

    if chisq <=0 or df < 1:
	return 1.0
    a = 0.5 * chisq
    if df%2 == 0:
	even = 1
    else:
	even = 0
    if df > 1:
	y = ex(-a)
    if even:
	s = y
    else:
	s = 2.0 * zprob(-math.sqrt(chisq))
    if (df > 2):
	chisq = 0.5 * (df - 1.0)
	if even:
	    z = 1.0
	else:
	    z = 0.5
	if a > BIG:
	    if even:
		e = 0.0
	    else:
		e = math.log(math.sqrt(math.pi))
	    c = math.log(a)
	    while (z <= chisq):
		e = math.log(z) + e
		s = s + ex(c*z-a-e)
		z = z + 1.0
	    return s
	else:
	    if even:
		e = 1.0
	    else:
		e = 1.0 / math.sqrt(math.pi) / math.sqrt(a)
		c = 0.0
		while (z <= chisq):
		    e = e * (a/float(z))
		    c = c + e
		    z = z + 1.0
		return (c*y+s)
    else:
	return s


def zprob(z):
    """
Returns the area under the normal curve 'to the left of' the given z value.
Thus, 
    for z<0, zprob(z) = 1-tail probability
    for z>0, 1.0-zprob(z) = 1-tail probability
    for any z, 2.0*(1.0-zprob(abs(z))) = 2-tail probability
Adapted from z.c in Gary Perlman's |Stat.

Usage:   zprob(z)
"""
    Z_MAX = 6.0    # maximum meaningful z-value
    if z == 0.0:
	x = 0.0
    else:
	y = 0.5 * math.fabs(z)
	if y >= (Z_MAX*0.5):
	    x = 1.0
	elif (y < 1.0):
	    w = y*y
	    x = ((((((((0.000124818987 * w
			-0.001075204047) * w +0.005198775019) * w
		      -0.019198292004) * w +0.059054035642) * w
		    -0.151968751364) * w +0.319152932694) * w
		  -0.531923007300) * w +0.797884560593) * y * 2.0
	else:
	    y = y - 2.0
	    x = (((((((((((((-0.000045255659 * y
			     +0.000152529290) * y -0.000019538132) * y
			   -0.000676904986) * y +0.001390604284) * y
			 -0.000794620820) * y -0.002034254874) * y
		       +0.006549791214) * y -0.010557625006) * y
		     +0.011630447319) * y -0.009279453341) * y
		   +0.005353579108) * y -0.002141268741) * y
		 +0.000535310849) * y +0.999936657524
    if z > 0.0:
	prob = ((x+1.0)*0.5)
    else:
	prob = ((1.0-x)*0.5)
    return prob


def fprob (dfnum, dfden, F):
    """
Returns the (1-tailed) significance level (p-value) of an F
statistic given the degrees of freedom for the numerator (dfR-dfF) and
the degrees of freedom for the denominator (dfF).

Usage:   fprob(dfnum, dfden, F)   where usually dfnum=dfbn, dfden=dfwn
"""
    p = betai(0.5*dfden, 0.5*dfnum, dfden/float(dfden+dfnum*F))
    return p


def betacf(a,b,x):
    """
This function evaluates the continued fraction form of the incomplete
Beta function, betai.  (Adapted from: Numerical Recipies in C.)

Usage:   betacf(a,b,x)
"""
    ITMAX = 200
    EPS = 3.0e-7

    bm = az = am = 1.0
    qab = a+b
    qap = a+1.0
    qam = a-1.0
    bz = 1.0-qab*x/qap
    for i in range(ITMAX+1):
	em = float(i+1)
	tem = em + em
	d = em*(b-em)*x/((qam+tem)*(a+tem))
	ap = az + d*am
	bp = bz+d*bm
	d = -(a+em)*(qab+em)*x/((qap+tem)*(a+tem))
	app = ap+d*az
	bpp = bp+d*bz
	aold = az
	am = ap/bpp
	bm = bp/bpp
	az = app/bpp
	bz = 1.0

	if (abs(az-aold)<(EPS*abs(az))):
	    return az
    print 'a or b too big, or ITMAX too small in Betacf.'
	
def betai(a,b,x):
    """
Returns the incomplete beta function:

    I-sub-x(a,b) = 1/B(a,b)*(Integral(0,x) of t^(a-1)(1-t)^(b-1) dt)

where a,b>0 and B(a,b) = G(a)*G(b)/(G(a+b)) where G(a) is the gamma
function of a.  The continued fraction formulation is implemented here,
using the betacf function.  (Adapted from: Numerical Recipies in C.)

Usage:   betai(a,b,x)
"""
    if (x<0.0 or x>1.0):
	raise ValueError, 'Bad x in lbetai'
    if (x==0.0 or x==1.0):
	bt = 0.0
    else:
	bt = math.exp(gammln(a+b)-gammln(a)-gammln(b)+a*math.log(x)+b*
		      math.log(1.0-x))
    if (x<(a+1.0)/(a+b+2.0)):
	return bt*betacf(a,b,x)/float(a)
    else:
    	return 1.0-bt*betacf(b,a,1.0-x)/float(b)
	

def tpvalue(t,df):
    """Returns the upper tail area for a t variate with df degrees of
    freedom.

    Arguments:
        t: t-statistic (scalar)
        df: degrees of freedom parameter

    Returns:
        pvalue: complement of the cdf for a t with df degrees of
        freedom.

    Author:
        Serge Rey based on modification of code in stats.py by 
        Gary Strangman.

    """

    prob = betai(0.5*df,0.5,float(df)/(df+t*t))
    return(prob)



def gammln(xx):
    """
Returns the gamma function of xx.
    Gamma(z) = Integral(0,infinity) of t^(z-1)exp(-t) dt.
(Adapted from: Numerical Recipies in C.)

Usage:   gammln(xx)
"""

    coeff = [76.18009173, -86.50532033, 24.01409822, -1.231739516,
	     0.120858003e-2, -0.536382e-5]
    x = xx - 1.0
    tmp = x + 5.5
    tmp = tmp - (x+0.5)*math.log(tmp)
    ser = 1.0
    for j in range(len(coeff)):
	x = x + 1
	ser = ser + coeff[j]/x
    return -tmp + math.log(2.50662827465*ser)



def dnorm(x,sigma=1,xbar=0):
    """evaluate normal distribution at x
    Arguments:
        x
        sigma: standard deviation
        xbar: mean value of x
    Returns:
        pdf: height of normal pdf at x
    
    Notes:
        original by Serge Rey <serge@rohan.sdsu.edu>
    
    """
    z=(x-xbar)/sigma
    zp = z > 20.
    zn = z < -20.
    zok = zp + zn

    z = ( zp * 20. ) + ( zn  *  -20. ) +  z * (zok == 0)
    try:
        return (1./((2*pi)**(1/2.)) * exp(-(1/2.) * z**2))
    except:
        print max(z),min(z)
        return z


