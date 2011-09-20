"""
tooltips.py: Contains tooltips for GeoDaSpace Preferences

The tips dictionary contains tooltips for wx widgets.  The dictionary key should be the name of the widget and the value the tooltip for that widget.
To add new tooltips simply find the widget ID in the XRC file for the widget you wish to identify.

Tooltips are added with the following logic...

Given:
window = preferencesDialog()
tips = {'CompInverse': "Warning for \"true inverse\": Slow Computation Times!"}

for widget in tips:
    if hasattr(window,widget):
        getattr(window,widget).SetToolTipText(tips[widget])
        getattr(window,widget+"Label").SetToolTipText(tips[widget])

notice that is the widget's Label is named "widgetLabel" the label will be given the tooltip as well.
"""
import preferences_xrc

tips = {
    'CompInverse': "Warning for \"true inverse\": Slow Computation Times!",
    'inferenceOnLambda': "Only affects models without heteroskedasticity. Based on Drukker, D.M., P. Egger, and I. R. Prucha. 2010. On two- step estimation of a spatial autoregressive model with autoregressive disturbances and endogenous regressors.  Technical report, Department of Economics, University of Maryland.",
    'OLSdiagnostics': "Diagnostics include tests for heteroskedasticity, normality, multi-collinearity, measures of fit, and other. Warning: Slow computation times for extremely large datasets.",
    'residualMoran': "Warning: Slower Computation Times!",
}
