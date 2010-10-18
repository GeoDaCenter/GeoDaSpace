#Script to test results of spError.GSLS against spdep's 

library(foreign)
library(spdep)

GMerrorsarMe <- function(#W, y, X, 
	formula, data = list(), listw, na.action=na.fail, 
	zero.policy=NULL, return_LL=FALSE, method="nlminb", 
        control=list(), pars, verbose=NULL, sparse_method="Matrix",
        returnHcov=FALSE, pWOrder=250, tol.Hcov=1.0e-10) {
#	ols <- lm(I(y) ~ I(X) - 1)
	mt <- terms(formula, data = data)
	mf <- lm(formula, data, na.action=na.action, method="model.frame")
	na.act <- attr(mf, "na.action")
	if (!is.null(na.act)) {
	    subset <- !(1:length(listw$neighbours) %in% na.act)
	    listw <- subset(listw, subset, zero.policy=zero.policy)
	}

	y <- model.extract(mf, "response")
	x <- model.matrix(mt, mf)

	# added aliased after trying boston with TOWN dummy
	lm.base <- lm(y ~ x - 1)
	aliased <- is.na(coefficients(lm.base))
	cn <- names(aliased)
	names(aliased) <- substr(cn, 2, nchar(cn))
	if (any(aliased)) {
		nacoef <- which(aliased)
		x <- x[,-nacoef]
	}
	ols <- lm(y ~ x - 1)
	if (missing(pars)) {
 	    ubase <- residuals(ols)
	    scorr <- c(crossprod(lag.listw(listw, ubase,
                zero.policy=zero.policy), ubase) / crossprod(ubase, ubase))
            scorr <- scorr / (sum(unlist(listw$weights)) / length(ubase))
            pars <- c(scorr, deviance(ols)/df.residual(ols))
        }
        if (length(pars) !=2 || !is.numeric(pars))
            stop("invalid starting parameter values")
	vv <- .kpwuwu(listw, residuals(ols), zero.policy=zero.policy)
#	nlsres <- nlm(.kpgm, pars, print.level=print.level, gradtol=gradtol, steptol=steptol, iterlim=iterlim, v=vv, verbose=verbose)
#	lambda <- nlsres$estimate[1]
        if (method == "nlminb")
            optres <- nlminb(pars, .kpgm, v=vv, verbose=verbose,
               control=control)
        else 
	    optres <- optim(pars, .kpgm, v=vv, verbose=verbose,
                method=method, control=control)
        if (optres$convergence != 0)
            warning(paste("convergence failure:", optres$message))
	lambda <- optres$par[1]
	names(lambda) <- "lambda"

	wy <- lag.listw(listw, y, zero.policy=zero.policy)
	if (any(is.na(wy)))
	    stop("NAs in lagged dependent variable")
	n <- NROW(x)
	m <- NCOL(x)
	xcolnames <- colnames(x)
	K <- ifelse(xcolnames[1] == "(Intercept)", 2, 1)
	if (any(is.na(wy)))
	    stop("NAs in lagged dependent variable")
	if (m > 1) {
	    WX <- matrix(nrow=n,ncol=(m-(K-1)))
	    for (k in K:m) {
		wx <- lag.listw(listw, x[,k], zero.policy=zero.policy)
		if (any(is.na(wx)))
		    stop("NAs in lagged independent variable")
		WX[,(k-(K-1))] <- wx
	    }
	}
	if (K == 2) {
# modified to meet other styles, email from Rein Halbersma
		wx1 <- as.double(rep(1, n))
		wx <- lag.listw(listw, wx1, zero.policy=zero.policy)
		if (m > 1) WX <- cbind(wx, WX)
		else WX <- matrix(wx, nrow=n, ncol=1)
	}
	colnames(WX) <- xcolnames
	rm(wx)
	lm.target <- lm(I(y - lambda*wy) ~ I(x - lambda*WX) - 1)
	r <- as.vector(residuals(lm.target))
	fit <- as.vector(y - r)
	p <- lm.target$rank
	SSE <- deviance(lm.target)
	s2 <- SSE/n
	rest.se <- (summary(lm.target)$coefficients[,2])*sqrt((n-p)/n)
	coef.lambda <- coefficients(lm.target)
	names(coef.lambda) <- xcolnames
	call <- match.call()
	names(r) <- names(y)
	names(fit) <- names(y)
	LL <- NULL
	if (return_LL) {
    		if (listw$style %in% c("W", "S") && !can.sim) {
			warning("No log likelihood value available")
		} else {
			if (sparse_method == "spam") {
                          if (!require(spam)) stop("spam not available")
			  if (listw$style %in% c("W", "S") & can.sim) {
			    csrw <- listw2U_spam(similar.listw_spam(listw))
			  } else csrw <- as.spam.listw(listw)
			  I <- diag.spam(1, n, n)
			} else if (sparse_method == "Matrix") {
			  if (listw$style %in% c("W", "S") & can.sim) {
			    csrw <- listw2U_Matrix(similar.listw_Matrix(listw))
			    similar <- TRUE
			  } else csrw <- as_dsTMatrix_listw(listw)
			  csrw <- as(csrw, "CsparseMatrix")
			  I <- as_dsCMatrix_I(n)
			} else stop("unknown sparse_method")
			gc(FALSE)
			yl <- y - lambda*wy
			xl <- x - lambda*WX
			xl.q <- qr.Q(qr(xl))
			xl.q.yl <- t(xl.q) %*% yl
			SSE <- t(yl) %*% yl - t(xl.q.yl) %*% xl.q.yl
			s2 <- SSE/n
            print(s2)
			if (sparse_method == "spam") {
			  Jacobian <- determinant((I - lambda * csrw), 
			    logarithm=TRUE)$modulus
			} else if (sparse_method == "Matrix") {
                             .f <- if (package_version(packageDescription(
                                 "Matrix")$Version) > "0.999375-30") 2 else 1
			  Jacobian <- .f * determinant(I - lambda * csrw,
 			    logarithm=TRUE)$modulus
			}
			gc(FALSE)
			LL <- (Jacobian -
				((n/2)*log(2*pi)) - (n/2)*log(s2) - 
				(1/(2*(s2)))*SSE)
		}
	}
        Hcov <- NULL
        if (returnHcov) {
            W <- as(as_dgRMatrix_listw(listw), "CsparseMatrix")
            pp <- ols$rank
            p1 <- 1L:pp
            R <- chol2inv(ols$qr$qr[p1, p1, drop = FALSE])
            B <- tcrossprod(R, x)
            B <- as(powerWeights(W=W, rho=lambda, order=pWOrder,
                X=B, tol=tol.Hcov), "matrix")
            C <- x %*% R
            C <- as(powerWeights(W=t(W), rho=lambda, order=pWOrder,
                X=C, tol=tol.Hcov), "matrix")
            Hcov <- B %*% C
            attr(Hcov, "method") <- "Matrix"
        }

	ret <- structure(list(lambda=lambda,
		coefficients=coef.lambda, rest.se=rest.se, 
		s2=s2, SSE=SSE, parameters=(m+2), lm.model=ols, 
		call=call, residuals=r, lm.target=lm.target,
		fitted.values=fit, formula=formula, aliased=aliased,
		zero.policy=zero.policy, LL=LL, vv=vv, optres=optres,
                pars=pars, Hcov=Hcov), class=c("gmsar"))

	if (!is.null(na.act))
		ret$na.action <- na.act
	ret
}

.kpwuwu <- function(W, u, zero.policy=FALSE) {
	n <- length(u)
# Gianfranco Piras 081119 
        trwpw <- sum(unlist(W$weights)^2)
#	tt <- matrix(0,n,1)
#	for (i in 1:n) {tt[i] <- sum(W$weights[[i]]^2) }
#	trwpw <- sum(tt)
	wu <- lag.listw(W, u, zero.policy=zero.policy)
	wwu <- lag.listw(W, wu, zero.policy=zero.policy)
    	uu <- crossprod(u,u)
    	uwu <- crossprod(u,wu)
    	uwpuw <- crossprod(wu,wu)
    	uwwu <- crossprod(u,wwu)
    	wwupwu <- crossprod(wwu,wu)
    	wwupwwu <- crossprod(wwu,wwu)
    	bigG <- matrix(0,3,3)
    	bigG[,1] <- c(2*uwu,2*wwupwu,(uwwu+uwpuw))/n
    	bigG[,2] <- - c(uwpuw,wwupwwu,wwupwu) / n
    	bigG[,3] <- c(1,trwpw/n,0)
    	litg <- c(uu,uwpuw,uwu) / n
    	list(bigG=bigG,litg=litg)
}

.kpgm <- function(rhopar,v,verbose=FALSE) {
  vv <- v$bigG %*% c(rhopar[1],rhopar[1]^2,rhopar[2]) - v$litg
  value <- sum(vv^2)
  value
  
}

dbf <- read.dbf('../../../trunk/econometrics/examples/columbus.dbf')
y <- dbf$HOVAL
x <- cbind(dbf$INC, dbf$CRIME)
w <- read.gal('../../../trunk/econometrics/examples/columbus.GAL')
w <- nb2listw(w)

modelMe <- GMerrorsarMe(y ~ INC + CRIME, data=dbf, w)
model <- GMerrorsar(y ~ INC + CRIME, data=dbf, w, return_LL=TRUE)

