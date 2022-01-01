import numpy as np
from .utils import custom_KDE



def log_pdf(m_list, pt, pts, weights, tf=True, clip=True):
    '''Compute log pdf error.

    This function is based on GPSEARCH.
    
    m_list: a list of Gaussian process regressor 
        The trained gpr used for prediction
    pt: an FFTKDE instance
        The true pdf
    pts: array like of shape (n,d)
        The input points used to compute pdf
    weights: array like of shape (n,)
        The pdf value of pts
    tf: bool 
        Whether m_list model is two-fidelity.
    '''
    res = np.zeros(len(m_list))

    for ii, model in enumerate(m_list):
        if tf:
            mu = np.empty(0)
            pts_aug = np.hstack((pts, np.ones((pts.shape[0],1))))
            pts_aug_list = np.array_split(pts_aug, 10)
            for iii in range(10):
                mu = np.concatenate((mu, 
                        model.predict(pts_aug_list[iii]).flatten()))
            #mu = model.predict(pts_aug).flatten()
        else:
            mu = np.empty(0)
            pts_list = np.array_split(pts, 10)
            for iii in range(10):
                mu = np.concatenate((mu, 
                        model.predict(pts_list[iii]).flatten()))
            #mu = model.predict(pts).flatten() 
        pb = custom_KDE(mu, weights=weights)

        x_min = min( pb.data.min(), pt.data.min() )
        x_max = max( pb.data.max(), pt.data.max() )
        rang = x_max-x_min
        x_eva = np.linspace(x_min - 0.01*rang,
                            x_max + 0.01*rang, 1024)

        yb, yt = pb.evaluate(x_eva), pt.evaluate(x_eva)
        log_yb, log_yt = np.log(yb), np.log(yt)

        if clip: # Clip to machine-precision
            np.clip(log_yb, -14, None, out=log_yb)
            np.clip(log_yt, -14, None, out=log_yt)

        log_diff = np.abs(log_yb-log_yt)
        noInf = np.isfinite(log_diff)
        res[ii] = np.trapz(log_diff[noInf], x_eva[noInf])

    return res


def log_pdf_(pt1, pt2, clip=True):
    x_min = min( pt1.data.min(), pt2.data.min() )
    x_max = max( pt1.data.max(), pt2.data.max() )
    rang = x_max-x_min
    x_eva = np.linspace(x_min - 0.01*rang,
                        x_max + 0.01*rang, 1024)

    yt1, yt2 = pt1.evaluate(x_eva), pt2.evaluate(x_eva)
    log_yt1, log_yt2 = np.log(yt1), np.log(yt2)

    if clip: # Clip to machine-precision
        np.clip(log_yt1, -14, None, out=log_yt1)
        np.clip(log_yt2, -14, None, out=log_yt2)

    log_diff = np.abs(log_yt1-log_yt2)
    noInf = np.isfinite(log_diff)
    res = np.trapz(log_diff[noInf], x_eva[noInf])

    return res



def failure_probability(m_list, pt, pts, weights, tf=True):
    '''Compute failure probability.
    '''
    res = np.zeros(len(m_list))
    for ii, model in enumerate(m_list):
        if tf:
            pts_aug = np.hstack((pts, np.ones((pts.shape[0],1))))
            mu = model.predict(pts_aug).flatten()
        else:
            mu = model.predict(pts).flatten() 
        res[ii] = abs(np.average(mu > 0, weights=weights) - pt)/pt
    return res


def moments_pdf(m_list, pts, weights, moment_true, tf=True, c_ratio=5, 
                power=6, center=0, 
                clip=True, accumulate=False):
    '''Compute the error in terms of high-order moments of the PDF.
    '''
    res = np.zeros(len(m_list))
    cost = np.zeros(len(m_list))

    for ii, model in enumerate(m_list):
        # compute the error
        if tf:
            pts_aug = np.hstack((pts, np.ones((pts.shape[0],1))))
            mu = model.predict(pts_aug).flatten() 
        else:
            mu = model.predict(pts).flatten() 
        moment_est = np.mean((mu-center)**power, weights)
        res[ii] = abs(moment_est - moment_true)
        # compute the computational cost
        X = np.copy(model.X_train_)
        if tf:
            num_h_X = np.count_nonzero(X[:,-1]==1)
            num_l_X = np.count_nonzero(X[:,-1]==0)
            cost[ii] = num_h_X + c_ratio * num_l_X
        else:
            cost[ii] = X.shape[0]

    if accumulate:
        res = np.minimum.accumulate(res)

    return cost, res

