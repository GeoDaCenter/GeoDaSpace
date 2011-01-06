import textwrap as TW
import numpy as np
import copy
import diagnostics
import diagnostics_sp






class DiagnosticBuilder:
    def __init__(self, x, constant, w, name_x, name_y, name_ds,\
                            vm, pred, name_yend=None, name_q=None,\
                            instruments=False):
        #general information
        self.r2 = diagnostics.r2(self)    
        self.ar2 = diagnostics.ar2(self)   
        self.sigML = self.sig2n  
        self.f_stat = diagnostics.f_stat(self)  
        self.logll = diagnostics.log_likelihood(self) 
        self.aic = diagnostics.akaike(self) 
        self.sc = diagnostics.schwarz(self) 
        
        #Coefficient, Std.Error, t-Statistic, Probability 
        self.std_err = diagnostics.se_betas(self)
        if instruments:
            self.z_stat = diagnostics.t_stat(self, z_stat=True)
        else:
            self.t_stat = diagnostics.t_stat(self)
        
        #part 2: REGRESSION DIAGNOSTICS 
        if instruments:
            self.mulColli = None
        else:
            self.mulColli = diagnostics.condition_index(self)
        self.jarque_bera = diagnostics.jarque_bera(self)
        
        #part 3: DIAGNOSTICS FOR HETEROSKEDASTICITY         
        self.breusch_pagan = diagnostics.breusch_pagan(self)
        self.koenker_bassett = diagnostics.koenker_bassett(self)
        if instruments:
            self.white = None
        else:
            self.white = diagnostics.white(self)
        
        #part 4: summary output
        if not name_x:
            name_x = ['var_'+str(i+1) for i in range(len(x[0]))]
        if constant:
            name_x.insert(0, 'CONSTANT')
        if not name_y:
            name_y = 'dep_var'
        if not name_ds:
            name_ds = 'unknown'
        if instruments:
            if not name_yend:
                self.name_yend = ['endogenous_'+str(i+1) for i in range(len(self.yend[0]))]
            else:
                self.name_yend = name_yend
            if not name_q:
                self.name_q = ['instrument_'+str(i+1) for i in range(len(self.q[0]))]
            else:
                self.name_q = name_q
            self.name_h = copy.copy(name_x)
            self.name_h.extend(self.name_q)
        self.name_x = name_x
        self.name_ds = name_ds
        self.name_y = name_y
        self.summary = summary_results(self, vm=vm, pred=pred, instruments=instruments)

        #part 5: spatial diagnostics
        if w:
            if instruments:
                ak = diagnostics_sp.ak_test(self, w)
                self.ak_test = (ak.mi, ak.ak, ak.p)
            else:
                lm_tests = diagnostics_sp.LMtests(self, w)
                self.lm_error = lm_tests.lme
                self.lm_lag = lm_tests.lml
                self.rlm_error = lm_tests.rlme
                self.rlm_lag = lm_tests.rlml
                self.lm_sarma = lm_tests.sarma
                moran_res = diagnostics_sp.MoranRes(self, w).I


def summary_results(reg, vm=False, pred=False, instruments=False):
    """
    nice output for regressions
    
    Parameters
    ----------

    reg     : regression object
              output instance from a regression model

    vm      : boolean
              if True, print out variance matrix

    pred    : boolean
              if True, print out y, predicted values and residuals
    
    Returns
    ----------

    strSummary   : string
                   formatted information from regression class

    """     
    strSummary = ""
    
    # general information 1
    strSummary += "REGRESSION\n"
    strSummary += "----------\n"
    title = "SUMMARY OF OUTPUT: " + reg.title + " ESTIMATION\n"
    strSummary += title
    strSummary += "-" * (len(title)-1) + "\n"
    strSummary += "%-20s:%12s\n" % ('Data set',reg.name_ds)
    strSummary += "%-20s:%12s  %-22s:%12d\n" % ('Dependent Variable',reg.name_y,'Number of Observations',reg.n)
    strSummary += "%-20s:%12.4f  %-22s:%12d\n" % ('Mean dependent var',reg.mean_y,'Number of Variables',reg.k)
    strSummary += "%-20s:%12.4f  %-22s:%12d\n" % ('S.D. dependent var',reg.std_y,'Degrees of Freedom',reg.n-reg.k)
    strSummary += '\n'

    # general information 2
    strSummary += "%-20s:%12.6f  %-22s:%12.4f\n" % ('R-squared',reg.r2,'F-statistic',reg.f_stat[0])
    strSummary += "%-20s:%12.6f  %-22s:%12.8g\n" % ('Adjusted R-squared',reg.ar2,'Prob(F-statistic)',reg.f_stat[1])
    strSummary += "%-20s:%12.3f  %-22s:%12.3f\n" % ('Sum squared residual',reg.utu,'Log likelihood',reg.logll)
    strSummary += "%-20s:%12.3f  %-22s:%12.3f\n" % ('Sigma-square',reg.sig2,'Akaike info criterion',reg.aic)
    strSummary += "%-20s:%12.3f  %-22s:%12.3f\n" % ('S.E. of regression',np.sqrt(reg.sig2),'Schwarz criterion',reg.sc)
    strSummary += "%-20s:%12.3f\n%-20s:%12.4f\n" % ('Sigma-square ML',reg.sigML,'S.E of regression ML',np.sqrt(reg.sigML))
    strSummary += '\n'
    
    # Variable    Coefficient     Std.Error    t-Statistic   Probability 
    strSummary += "----------------------------------------------------------------------------\n"
    if instruments:
        strSummary += "    Variable     Coefficient       Std.Error     z-Statistic     Probability\n"
    else:
        strSummary += "    Variable     Coefficient       Std.Error     t-Statistic     Probability\n"
    strSummary += "----------------------------------------------------------------------------\n"
    i = 0
    if instruments:
        for name in reg.name_x:        
            print "exog", i, name
            strSummary += "%12s    %12.7f    %12.7f    %12.7f    %12.7g\n" % (name,reg.betas[i][0],reg.std_err[i],reg.z_stat[i][0],reg.z_stat[i][1])
            i += 1
        for name in reg.name_yend:        
            print "endog", i, name
            strSummary += "%12s    %12.7f    %12.7f    %12.7f    %12.7g\n" % (name,reg.betas[i][0],reg.std_err[i],reg.z_stat[i][0],reg.z_stat[i][1])
            i += 1
        strSummary += "----------------------------------------------------------------------------\n"
        insts = "Instruments: "
        for name in reg.name_h:
            insts += name + ", "
        text_wrapper = TW.TextWrapper(width=76, subsequent_indent="             ")
        insts = text_wrapper.fill(insts[:-2])
        strSummary += insts + "\n"
    else:
        for name in reg.name_x:        
            strSummary += "%12s    %12.7f    %12.7f    %12.7f    %12.7g\n" % (name,reg.betas[i][0],reg.std_err[i],reg.t_stat[i][0],reg.t_stat[i][1])
            i += 1
        strSummary += "----------------------------------------------------------------------------\n"
    
    # diagonostics
    strSummary += "\n\nREGRESSION DIAGNOSTICS\n"
    if reg.mulColli:
        strSummary += "MULTICOLLINEARITY CONDITION NUMBER%12.6f\n" % (reg.mulColli)
    strSummary += "TEST ON NORMALITY OF ERRORS\n"
    strSummary += "TEST                  DF          VALUE            PROB\n"
    strSummary += "%-22s%2d       %12.6f        %9.7f\n\n" % ('Jarque-Bera',reg.jarque_bera['df'],reg.jarque_bera['jb'],reg.jarque_bera['pvalue'])
    strSummary += "DIAGNOSTICS FOR HETEROSKEDASTICITY\n"
    strSummary += "RANDOM COEFFICIENTS\n"
    strSummary += "TEST                  DF          VALUE            PROB\n"
    strSummary += "%-22s%2d       %12.6f        %9.7f\n" % ('Breusch-Pagan test',reg.breusch_pagan['df'],reg.breusch_pagan['bp'],reg.breusch_pagan['pvalue'])
    strSummary += "%-22s%2d       %12.6f        %9.7f\n" % ('Koenker-Bassett test',reg.koenker_bassett['df'],reg.koenker_bassett['kb'],reg.koenker_bassett['pvalue'])
    if reg.white:
        strSummary += "SPECIFICATION ROBUST TEST\n"
        strSummary += "TEST                  DF          VALUE            PROB\n"
        strSummary += "%-22s%2d       %12.6f        %9.7f\n\n" % ('White',reg.white['df'],reg.white['wh'],reg.white['pvalue'])

    # variance matrix
    if vm:
        strVM = ""
        strVM += "COEFFICIENTS VARIANCE MATRIX\n"
        strVM += "----------------------------\n"
        strVM += "%12s" % ('CONSTANT')
        for name in reg.name_x:
            strVM += "%12s" % (name)
        strVM += "\n"
        nrow = reg.vm.shape[0]
        ncol = reg.vm.shape[1]
        for i in range(nrow):
            for j in range(ncol):
                strVM += "%12.6f" % (reg.vm[i][j]) 
            strVM += "\n"
        strSummary += strVM
        
    # y, PREDICTED, RESIDUAL 
    if pred:
        strPred = "\n\n"
        strPred += "%16s%16s%16s%16s\n" % ('OBS',reg.name_y,'PREDICTED','RESIDUAL')
        for i in range(reg.n):
            strPred += "%16d%16.5f%16.5f%16.5f\n" % (i+1,reg.y[i][0],reg.predy[i][0],reg.u[i][0])
        strSummary += strPred
            
    # end of report
    strSummary += "========================= END OF REPORT =============================="
        
    return strSummary


