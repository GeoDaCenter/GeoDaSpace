import textwrap as TW
import numpy as np
import diagnostics


class Diagnostic_Builder:
    def __init__(self, constant, vm, pred, instruments=False):
        #general information
        self.r2 = diagnostics.r2(self)    
        self.ar2 = diagnostics.ar2(self)   
        self.sigML = self.sig2  
        self.Fstat = diagnostics.f_stat(self)  
        self.logll = diagnostics.log_likelihood(self) 
        self.aic = diagnostics.akaike(self) 
        self.sc = diagnostics.schwarz(self) 
        
        #Coefficient, Std.Error, t-Statistic, Probability 
        self.std_err = diagnostics.se_betas(self)
        self.Tstat = diagnostics.t_stat(self)
        
        #part 2: REGRESSION DIAGNOSTICS 
        self.mulColli = diagnostics.condition_index(self)
        self.diag = {}
        self.diag['JB'] = diagnostics.jarque_bera(self)
        
        #part 3: DIAGNOSTICS FOR HETEROSKEDASTICITY         
        self.diag['BP'] = diagnostics.breusch_pagan(self)
        self.diag['KB'] = {'df':2,'kb':5.694088,'pvalue':0.0580156}
        self.diag['WH'] = {'df':5,'wh':19.94601,'pvalue':0.0012792}
        
        #part 4: summary output
        self.summary = summary_results(self, constant=constant, vm=vm, pred=pred, instruments=instruments)


def summary_results(reg, constant=True, vm = False, pred = False, instruments=False):
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
    strSummary += "%-20s:%12.6f  %-22s:%12.4f\n" % ('R-squared',reg.r2,'F-statistic',reg.Fstat[0])
    strSummary += "%-20s:%12.6f  %-22s:%12.8g\n" % ('Adjusted R-squared',reg.ar2,'Prob(F-statistic)',reg.Fstat[1])
    strSummary += "%-20s:%12.3f  %-22s:%12.3f\n" % ('Sum squared residual',reg.utu,'Log likelihood',reg.logll)
    strSummary += "%-20s:%12.3f  %-22s:%12.3f\n" % ('Sigma-square',reg.sig2,'Akaike info criterion',reg.aic)
    strSummary += "%-20s:%12.3f  %-22s:%12.3f\n" % ('S.E. of regression',np.sqrt(reg.sig2),'Schwarz criterion',reg.sc)
    strSummary += "%-20s:%12.3f\n%-20s:%12.4f\n" % ('Sigma-square ML',reg.sigML,'S.E of regression ML',np.sqrt(reg.sigML))
    strSummary += '\n'
    
    # Variable    Coefficient     Std.Error    t-Statistic   Probability 
    strSummary += "----------------------------------------------------------------------------\n"
    strSummary += "    Variable     Coefficient       Std.Error     t-Statistic     Probability\n"
    strSummary += "----------------------------------------------------------------------------\n"
    if constant:
        strSummary += "%12s    %12.7f    %12.7f    %12.7f    %12.7g\n" % ('CONSTANT',reg.betas[0][0],reg.std_err[0],reg.Tstat[0][0],reg.Tstat[0][1])
        i = 1
    else:
        i = 0
    for name in reg.name_x:        
        strSummary += "%12s    %12.7f    %12.7f    %12.7f    %12.7g\n" % (name,reg.betas[i][0],reg.std_err[i],reg.Tstat[i][0],reg.Tstat[i][1])
        i += 1
    strSummary += "----------------------------------------------------------------------------\n"
    if instruments:
        insts = "Instruments: "
        for name in reg.name_h:
            insts += name + ", "
        text_wrapper = TW.TextWrapper(width=76, subsequent_indent="             ")
        insts = text_wrapper.fill(insts[:-2])
        strSummary += insts + "\n"
    
    # diagonostics
    strSummary += "\n\nREGRESSION DIAGNOSTICS\n"
    strSummary += "MULTICOLLINEARITY CONDITION NUMBER%12.6f\n" % (reg.mulColli)
    strSummary += "TEST ON NORMALITY OF ERRORS\n"
    strSummary += "TEST                  DF          VALUE            PROB\n"
    strSummary += "%-22s%2d       %12.6f        %9.7f\n\n" % ('Jarque-Bera',reg.diag['JB']['df'],reg.diag['JB']['jb'],reg.diag['JB']['pvalue'])
    strSummary += "DIAGNOSTICS FOR HETEROSKEDASTICITY\n"
    strSummary += "RANDOM COEFFICIENTS\n"
    strSummary += "TEST                  DF          VALUE            PROB\n"
    strSummary += "%-22s%2d       %12.6f        %9.7f\n" % ('Breusch-Pagan test',reg.diag['BP']['df'],reg.diag['BP']['bp'],reg.diag['BP']['pvalue'])
    strSummary += "%-22s%2d       %12.6f        %9.7f\n" % ('Koenker-Bassett test',reg.diag['KB']['df'],reg.diag['KB']['kb'],reg.diag['KB']['pvalue'])
    strSummary += "SPECIFICATION ROBUST TEST\n"
    strSummary += "TEST                  DF          VALUE            PROB\n"
    strSummary += "%-22s%2d       %12.6f        %9.7f\n\n" % ('White',reg.diag['WH']['df'],reg.diag['WH']['wh'],reg.diag['WH']['pvalue'])

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


