Search.setIndex({"docnames": ["api", "capabilities", "credits", "index", "install", "notebooks/covariance_propagation", "notebooks/covariance_transformation", "notebooks/gradient_based_optimization", "notebooks/sgp4_partial_derivatives", "notebooks/tle_object", "notebooks/tle_propagation", "tutorials"], "filenames": ["api.rst", "capabilities.ipynb", "credits.ipynb", "index.md", "install.rst", "notebooks/covariance_propagation.ipynb", "notebooks/covariance_transformation.ipynb", "notebooks/gradient_based_optimization.ipynb", "notebooks/sgp4_partial_derivatives.ipynb", "notebooks/tle_object.ipynb", "notebooks/tle_propagation.ipynb", "tutorials.rst"], "titles": ["API", "Capabilities", "Credits", "<span class=\"math notranslate nohighlight\">\\(\\partial\\textrm{SGP4}\\)</span> Documentation", "Installation", "Covariance Propagation", "Similarity Transformation - from Cartesian to TLE Covariance", "Gradient Based Optimization", "Partial Derivatives Computation via Autodiff", "TLE Object", "Propagate TLEs", "Tutorials"], "terms": {"partialtextrm": 0, "sgp4": [0, 1, 2, 6, 7, 8, 10], "dsgp4": [1, 3, 4, 5, 6, 7, 8, 9, 10, 11], "i": [1, 3, 4, 5, 6, 7, 8, 10], "an": [1, 4, 5, 6, 7], "open": [1, 4], "sourc": 1, "project": [1, 2, 4], "constitut": 1, "differenti": [1, 3, 6, 7, 8], "version": 1, "partial": [2, 5, 6, 7, 10, 11], "textrm": [2, 5, 6, 7, 8, 10], "wa": [2, 4, 6, 9], "develop": [2, 3, 4], "dure": [2, 10], "sponsor": 2, "univers": 2, "oxford": 2, "while": 2, "giacomo": [2, 3], "acciarini": [2, 3], "ox4ailab": 2, "collabor": 2, "dr": 2, "at\u0131l\u0131m": [2, 3], "g\u00fcne\u015f": [2, 3], "baydin": [2, 3], "The": [2, 3, 8, 9], "main": [2, 3], "ar": [2, 3, 6, 7, 8, 10, 11], "gmail": [2, 3], "com": [2, 3, 4], "gune": 2, "robot": 2, "ox": 2, "ac": 2, "uk": 2, "program": 3, "written": [3, 8], "leverag": [3, 6, 11], "pytorch": [3, 8], "machin": 3, "learn": 3, "framework": [3, 11], "thi": [3, 4, 5, 6, 7, 8, 9, 10], "enabl": 3, "featur": [3, 8], "like": [3, 10], "automat": [3, 8], "batch": [3, 5], "propag": [3, 6, 7, 8, 9, 11], "across": 3, "differ": [3, 7], "tle": [3, 5, 7, 11], "were": 3, "previous": 3, "avail": [3, 4, 5, 6], "origin": [3, 8], "implement": [3, 8], "author": 3, "dario": 3, "izzo": 3, "instal": 3, "capabl": 3, "credit": 3, "tutori": [3, 6, 8, 9], "api": [3, 8, 9, 10], "ha": [4, 9, 10], "follow": [4, 7, 8, 9], "python": 4, "first": [4, 5, 6, 8, 9, 10], "we": [4, 5, 6, 7, 8, 9, 10], "add": 4, "forg": 4, "channel": 4, "config": 4, "set": [4, 9], "channel_prior": 4, "strict": 4, "now": [4, 5, 6, 7, 8, 9], "can": [4, 5, 6, 7, 8, 9, 10], "either": [4, 9], "through": 4, "mamba": 4, "pypi": 4, "http": 4, "org": 4, "you": 4, "via": [4, 7, 11], "us": [4, 5, 6, 8, 9, 10, 11], "git": 4, "clone": 4, "github": 4, "esa": 4, "usual": [4, 5], "pr": 4, "base": [4, 11], "workflow": 4, "thu": 4, "": [4, 5, 6, 8, 9, 10], "master": 4, "branch": 4, "normal": 4, "kept": 4, "work": [4, 8], "state": [4, 5, 6, 7, 8], "successfulli": 4, "compil": 4, "run": 4, "test": 4, "To": 4, "do": [4, 5, 6, 7], "so": [4, 7], "must": 4, "option": 4, "pytest": 4, "If": [4, 5, 6, 10], "command": 4, "execut": 4, "without": 4, "ani": [4, 7], "error": 4, "your": 4, "readi": 4, "troubl": 4, "pleas": 4, "hesit": 4, "contact": 4, "u": [4, 5, 6, 9, 10], "issu": 4, "report": 4, "In": [5, 6, 7, 8, 9, 10], "notebook": [5, 6, 8, 9, 10], "discuss": [5, 6, 9], "how": [5, 6, 8, 9, 10, 11], "appli": [5, 6, 9], "order": [5, 7], "approxim": [5, 7], "matrix": [5, 6, 8], "begin": [5, 6, 7, 8], "equat": [5, 6, 7, 8], "p_": [5, 6], "pmb": [5, 8], "x": [5, 7, 8, 10], "_f": 5, "dfrac": [5, 6, 7, 8], "_0": [5, 7], "_": [5, 7], "0": [5, 6, 7, 8, 9, 10], "t": [5, 6, 7, 8], "text": [5, 6], "end": [5, 6, 7, 8], "where": [5, 6, 7, 8], "initi": [5, 6, 7, 8, 10], "time": [5, 6, 7, 9, 10], "t_0": [5, 7], "vector": [5, 7, 8], "cartesian": [5, 7, 11], "teme": [5, 6], "certain": [5, 6], "t_f": 5, "import": [5, 6, 7, 8, 10], "aspect": 5, "notic": 5, "abov": [5, 6, 9], "formula": 5, "besid": 5, "also": [5, 8, 9, 10], "transform": [5, 11], "from": [5, 7, 11], "paramet": [5, 6], "ones": 5, "let": [5, 6, 8, 9, 10], "torch": [5, 6, 8, 10], "numpi": [5, 6, 9], "np": [5, 6], "And": 5, "construct": [5, 6, 8, 9, 10], "some": [5, 11], "object": [5, 8, 11], "inp_fil": [5, 8], "pslv": [5, 8], "deb": [5, 6, 7, 8, 9], "1": [5, 6, 7, 8, 9, 10], "35351u": [5, 8], "01049qk": [5, 8], "22066": [5, 8], "70636923": [5, 8], "00002156": [5, 8], "00000": [5, 6, 7, 8, 9], "63479": [5, 8], "3": [5, 6, 7, 8, 9, 10], "9999": [5, 6, 8], "2": [5, 6, 7, 8, 9, 10], "35351": [5, 8], "98": [5, 8], "8179": [5, 8], "29": [5, 8], "5651": [5, 8], "0005211": [5, 8], "45": [5, 8], "5944": [5, 8], "314": [5, 8], "5671": [5, 8], "14": [5, 6, 7, 8, 9], "44732274457505": [5, 8], "cosmo": [5, 6, 7, 8, 9], "2251": [5, 6, 7, 8, 9], "34550u": 5, "93036ul": 5, "22068": [5, 6, 7, 8, 9], "48847750": 5, "00005213": 5, "16483": 5, "9994": [5, 9], "34550": 5, "73": 5, "9920": 5, "106": 5, "8624": 5, "0177754": 5, "242": 5, "5189": 5, "115": 5, "7794": 5, "31752709674389": 5, "ggse": 5, "1292u": 5, "65016c": 5, "22069": 5, "19982496": 5, "00000050": 5, "68899": 5, "4": [5, 6, 7, 8, 9, 10], "9998": [5, 8], "1292": 5, "70": 5, "0769": 5, "62": 5, "6868": 5, "0022019": 5, "223": 5, "3241": 5, "136": 5, "6137": 5, "00692829906555": 5, "cz": 5, "2d": 5, "35275u": 5, "08056e": 5, "48794021": 5, "00000595": 5, "24349": 5, "35275": 5, "8424": 5, "84": [5, 9], "5826": 5, "0139050": 5, "155": 5, "2017": 5, "205": 5, "5930": 5, "25810852685662": 5, "fengyun": 5, "1c": 5, "35237u": 5, "99025dqv": 5, "17158141": 5, "00009246": 5, "14994": 5, "9992": [5, 8, 9], "35237": 5, "99": [5, 8], "5946": 5, "147": 5, "0483": 5, "0055559": 5, "41": 5, "0775": 5, "319": 5, "4589": 5, "70446744775205": 5, "splitlin": [5, 8], "tsinc": [5, 8, 10], "linspac": [5, 10], "5": [5, 6, 7, 8, 10], "60": [5, 10], "24": [5, 10], "8": [5, 6, 8], "dai": [5, 9, 10], "index": 5, "rang": [5, 6, 8], "len": [5, 8, 10], "k": [5, 7, 8], "append": [5, 6, 8, 9], "repeat": 5, "track": [5, 8, 9], "gradient": [5, 8, 11], "w": [5, 6, 7, 8], "tle_el": [5, 6, 8], "initialize_tl": [5, 6, 7, 8, 10], "with_grad": [5, 6, 8], "true": [5, 6, 8, 10], "orbit": [5, 10], "state_tem": [5, 6, 8, 10], "propagate_batch": [5, 8, 10], "output": [5, 8], "r": [5, 6, 7, 8], "build": [5, 8], "deriv": [5, 9, 11], "shape": [5, 8], "nx6x9": [5, 8], "n": [5, 8, 10], "number": [5, 8, 9], "element": [5, 6, 7, 8, 9, 10], "6": [5, 6, 8, 10], "9": [5, 6, 8, 10], "dx_dtle": [5, 6], "zero": [5, 6, 7, 8], "grad": [5, 6, 8], "none": [5, 6, 8], "flatten": [5, 6, 8], "backward": [5, 6, 8], "retain_graph": [5, 6, 8], "defin": [5, 7, 9], "cov_xyz": 5, "cov_rtn": 5, "cov_tl": [5, 6], "arrai": [5, 6], "06817079e": 5, "23": 5, "85804989e": 5, "25": 5, "51328946e": 5, "48167092e": 5, "13": [5, 6, 7, 8], "80784129e": 5, "11": [5, 9], "17516946e": 5, "17": [5, 8], "80719145e": 5, "47782854e": 5, "06374440e": 5, "19": 5, "10888880e": 5, "26": 5, "19571327e": 5, "28": 5, "57097512e": 5, "27": 5, "57618033e": 5, "16": 5, "87675528e": 5, "28440729e": 5, "20": [5, 7], "87608028e": 5, "57219640e": 5, "10539220e": 5, "22": [5, 7], "62208982e": 5, "00370060e": 5, "13145502e": 5, "41515501e": 5, "13025898e": 5, "12": [5, 7, 10], "11161104e": 5, "12805538e": 5, "40212641e": 5, "15": [5, 9], "61913778e": 5, "72347076e": 5, "25849762e": 5, "85837006e": 5, "69613552e": 5, "03": [5, 6, 8, 10], "60858400e": 5, "01": [5, 6, 8, 10], "44529381e": 5, "06": [5, 8], "60848714e": 5, "66633934e": 5, "04": [5, 6, 8, 10], "73554382e": 5, "10": [5, 6, 8], "98398791e": 5, "64526723e": 5, "81073778e": 5, "35783115e": 5, "70385711e": 5, "35662070e": 5, "60149046e": 5, "02": [5, 6, 8, 10], "68286334e": 5, "07": [5, 8], "00692291e": 5, "34937429e": 5, "18": [5, 8], "42650203e": 5, "75093676e": 5, "05": [5, 6, 8], "70379140e": 5, "42796071e": 5, "62336921e": 5, "98327475e": 5, "64467582e": 5, "80972744e": 5, "35541747e": 5, "60131157e": 5, "28248504e": 5, "71925407e": 5, "25500097e": 5, "85239630e": 5, "63783423e": 5, "21657063e": 5, "16764843e": 5, "73507662e": 5, "21": 5, "65971417e": 5, "73554368e": 5, "68286335e": 5, "28248505e": 5, "21656925e": 5, "21029182e": 5, "1000": [5, 7], "frob_norm": 5, "def": [5, 6], "rotation_matrix": [5, 6], "comput": [5, 6, 7, 11], "uvw": [5, 6], "rotat": [5, 6], "arg": [5, 6], "row": [5, 6], "column": [5, 6], "repres": [5, 6], "posit": [5, 6, 7, 8], "second": [5, 6, 8, 9], "veloc": [5, 6, 7, 8], "return": [5, 6, 7, 8], "v": [5, 6], "linalg": [5, 6], "norm": [5, 6], "cross": [5, 6], "vstack": [5, 6], "from_cartesian_to_rtn": [5, 6], "cartesian_to_rtn_rotation_matrix": [5, 6], "convert": [5, 6], "rtn": [5, 6], "frame": [5, 6], "suppli": [5, 6], "otherwis": [5, 6, 10], "r_rtn": [5, 6], "dot": [5, 6, 7, 8], "v_rtn": [5, 6], "stack": [5, 6], "final": [5, 8], "all": [5, 8, 9, 10], "idx": 5, "enumer": 5, "state_rtn": [5, 6], "detach": [5, 6], "6x6": 5, "transformation_matrix_cartesian_to_rtn": [5, 6], "matmul": [5, 6], "ord": 5, "fro": 5, "plot": [5, 10], "frobeniu": 5, "each": [5, 8, 10], "function": [5, 7], "matplotlib": [5, 10], "pyplot": 5, "plt": 5, "32": [5, 8], "40": 5, "semilogi": 5, "label": [5, 10], "line0": 5, "legend": 5, "xlabel": [5, 10], "min": [5, 9], "ylabel": [5, 10], "One": [6, 8], "obviou": 6, "applic": [6, 8], "its": [6, 7], "see": [6, 7, 8, 9], "covariance_propag": [6, 8], "ipynb": 6, "case": [6, 8, 10], "start": 6, "correspond": [6, 7], "given": [6, 7, 10], "refer": 6, "Then": [6, 9], "assum": [6, 9, 10], "have": [6, 7, 8, 9, 10], "associ": 6, "instanc": [6, 9], "fundament": 6, "astrodynam": 6, "vallado": 6, "theoret": 6, "background": 6, "want": [6, 7, 8, 10], "coordin": 6, "y": [6, 7, 8, 10], "m": [6, 8], "p_x": 6, "m_": 6, "ij": [6, 7], "y_i": [6, 7], "x_j": 6, "onc": 6, "done": [6, 7], "left": 6, "directli": [6, 9, 10], "which": [6, 7, 8, 10], "sever": 6, "exampl": [6, 7, 8, 9, 10, 11], "gener": [6, 7], "perturb": 6, "mean": [6, 7, 8, 9], "feed": 6, "them": [6, 8, 10], "noisi": 6, "observ": [6, 7], "As": [6, 8], "alwai": [6, 8, 10], "line": [6, 8, 9, 10], "34427u": 6, "93036ru": 6, "94647328": 6, "00008100": 6, "11455": 6, "34427": 6, "74": [6, 7, 8, 9], "0145": 6, "306": 6, "8269": 6, "0033346": 6, "0723": 6, "347": 6, "1308": 6, "76870515693886": 6, "my_tl": [6, 7, 10], "matrix_rtn": 6, "cov": 6, "cov_matrix_rtn": 6, "sigma_rr": 6, "sigma_rt": 6, "sigma_rn": 6, "sigma_vr": 6, "sigma_vt": 6, "sigma_vn": 6, "diag": 6, "covariance_diagon": 6, "100": [6, 7, 8], "1e": [6, 7, 9], "print": [6, 7, 8, 9], "f": [6, 7, 9, 10], "e": [6, 7, 8], "00": [6, 8, 10], "tensor": [6, 8, 10], "1941e": 6, "6009e": 6, "2616e": 6, "6578e": 6, "2582e": 6, "7": [6, 7, 8], "2697e": 6, "grad_fn": [6, 8], "transposebackward0": [6, 8], "c_teme": 6, "43678346e": 6, "27091412e": 6, "90611714e": 6, "00000000e": 6, "24494699e": 6, "42742252e": 6, "31826956e": 6, "80461804e": 6, "59800553e": 6, "09672885e": 6, "69441685e": 6, "57016478e": 6, "25009651e": 6, "quick": 6, "check": 6, "confirm": 6, "correct": 6, "c_rtn2": 6, "allclos": 6, "For": [6, 10], "detail": 6, "out": 6, "tle_propag": [6, 8], "6x9": 6, "screen": [6, 8], "0000e": [6, 8, 10], "7924e": 6, "5394e": 6, "0479e": [6, 8], "5463e": 6, "3390e": 6, "9523e": 6, "1618e": 6, "3683e": 6, "1736e": 6, "7875e": 6, "0267e": 6, "7277e": 6, "9559e": 6, "7716e": 6, "4301e": 6, "6362e": 6, "5385e": 6, "8152e": 6, "5521e": 6, "6044e": 6, "3649e": 6, "0581e": [6, 8], "3574e": 6, "0789e": 6, "4300e": 6, "0851e": 6, "4891e": 6, "0766e": 6, "7732e": 6, "7622e": 6, "obtain": [6, 7], "9x9": 6, "pinv": 6, "thing": [6, 8], "correctli": [6, 7], "c_xyz_2": 6, "call": [7, 10], "look": 7, "futur": [7, 8, 10], "t_": 7, "ob": 7, "rightarrow": 7, "vec": 7, "z": [7, 8, 10], "find": 7, "That": 7, "when": 7, "abl": 7, "invert": 7, "formul": 7, "minimum": 7, "free": 7, "variabl": 7, "between": 7, "make": [7, 10], "current": 7, "reformul": 7, "align": 7, "minim": 7, "tild": 7, "newton": 7, "method": [7, 9, 10], "updat": [7, 9], "guess": 7, "y_": 7, "until": 7, "converg": 7, "df": 7, "y_k": 7, "jacobian": [7, 8], "respect": 7, "easili": 7, "made": [7, 8], "df_": 7, "j": [7, 8], "_1": 7, "_2": 7, "_3": 7, "_4": 7, "_5": 7, "_6": 7, "no_": 7, "kozai": 7, "ecco": 7, "inclo": 7, "mo": 7, "argpo": 7, "nodeo": 7, "n_": 7, "ddot": [7, 8], "b": [7, 8], "sinc": [7, 8], "built": 7, "input": 7, "quit": 7, "furthermor": 7, "found": 7, "simpl": [7, 11], "invers": 7, "keplerian": 7, "doe": [7, 10], "good": 7, "load": [7, 8, 10], "file_nam": 7, "extract": [7, 9, 10], "one": [7, 10], "expect": 7, "ident": [7, 8], "found_tl": 7, "newton_method": 7, "tle_0": 7, "time_mjd": 7, "date_mjd": 7, "new_tol": 7, "max_it": 7, "7582159128252637e": 7, "solut": 7, "iter": 7, "34454u": [7, 8, 9], "93036sx": [7, 8, 9], "91971155": [7, 8, 9], "00000319": [7, 8, 9], "11812": [7, 8, 9], "9996": [7, 8, 9], "34454": [7, 8, 9], "0583": [7, 8, 9], "280": [7, 8, 9], "7094": [7, 8, 9], "0037596": [7, 8, 9], "327": [7, 8, 9], "9100": [7, 8, 9], "31": [7, 8, 9], "9764": [7, 8, 9], "35844873683320": [7, 8, 9], "91971967": 7, "9997": [7, 9], "minut": [7, 8], "after": 7, "800": 7, "still": 7, "cours": 7, "547518096460396e": 7, "24138": 7, "254": 7, "2494": 7, "0037442": 7, "103": 7, "1744": 7, "5962": 7, "36399602683320": 7, "show": [8, 9, 10], "due": 8, "fact": 8, "support": 8, "autograd": 8, "more": [8, 11], "advanc": 8, "practic": 8, "state_transition_matrix_comput": 8, "graident_based_optim": 8, "orbit_determin": 8, "creat": [8, 9], "shown": 8, "howev": 8, "instead": 8, "standard": 8, "requir": [8, 10], "record": 8, "oper": 8, "variou": 8, "take": 8, "random": 8, "rand": 8, "requires_grad": 8, "fals": [8, 10], "4315e": 8, "9748e": 8, "8700e": 8, "9401e": 8, "8861e": 8, "1914e": 8, "4174e": 8, "9803e": 8, "3495e": 8, "9515e": 8, "3302e": 8, "1943e": 8, "3724e": 8, "9950e": 8, "7043e": 8, "9866e": 8, "5719e": 8, "2006e": 8, "3746e": 8, "9944e": 8, "7841e": 8, "9849e": 8, "6572e": 8, "2004e": 8, "3806e": 8, "9927e": 8, "9994e": 8, "8874e": 8, "1998e": 8, "4096e": 8, "9832e": 8, "0594e": 8, "9577e": 8, "0204e": 8, "1957e": 8, "3650e": 8, "9970e": 8, "4356e": 8, "9922e": 8, "2845e": 8, "2013e": 8, "4205e": 8, "9791e": 8, "4626e": 8, "9490e": 8, "4510e": 8, "1937e": 8, "3690e": 8, "9960e": 8, "5805e": 8, "9891e": 8, "4394e": 8, "2009e": 8, "4342e": 8, "9737e": 8, "9699e": 8, "9379e": 8, "1908e": 8, "retriev": 8, "v_x": 8, "v_y": 8, "v_z": 8, "type": 8, "d": 8, "dx": 8, "dt": 8, "dy": 8, "dz": 8, "2x": 8, "2y": 8, "2z": 8, "dv_x": 8, "dv_y": 8, "dv_z": 8, "care": 8, "about": 8, "mirror": 8, "km": [8, 9, 10], "henc": 8, "dimens": 8, "coher": 8, "si": 8, "convers": 8, "partial_deriv": 8, "zeros_lik": 8, "ones_lik": 8, "1641e": 8, "7317e": 8, "3148e": 8, "4548e": 8, "6066e": 8, "5627e": 8, "1709e": 8, "3982e": 8, "3166e": 8, "3620e": 8, "6104e": 8, "2181e": 8, "1919e": 8, "3431e": 8, "3204e": 8, "0656e": 8, "6206e": 8, "1287e": 8, "1909e": 8, "3944e": 8, "3203e": 8, "0801e": 8, "6201e": 8, "1816e": 8, "1882e": 8, "5325e": 8, "3199e": 8, "1191e": 8, "6189e": 8, "3241e": 8, "1747e": 8, "2122e": 8, "3174e": 8, "3102e": 8, "6124e": 8, "0260e": 8, "1953e": 8, "1707e": 8, "3208e": 8, "0167e": 8, "6220e": 8, "5081e": 8, "1694e": 8, "4706e": 8, "3162e": 8, "3822e": 8, "6096e": 8, "2929e": 8, "1935e": 8, "2637e": 8, "3206e": 8, "0431e": 8, "6212e": 8, "0467e": 8, "1628e": 8, "7957e": 8, "3145e": 8, "4725e": 8, "6059e": 8, "6288e": 8, "basic": 8, "35350u": 8, "01049qj": 8, "76869562": 8, "00000911": 8, "24939": 8, "35350": 8, "6033": 8, "64": 8, "7516": 8, "0074531": 8, "8340": 8, "261": 8, "1278": 8, "48029442457561": 8, "sl": 8, "35354u": 8, "93014bd": 8, "76520028": 8, "00021929": 8, "20751": 8, "9995": 8, "35354": 8, "75": 8, "7302": 8, "7819": 8, "0059525": 8, "350": 8, "7978": 8, "2117": 8, "92216400847487": 8, "35359u": 8, "93014bj": 8, "55187275": 8, "00025514": 8, "24908": 8, "35359": 8, "7369": 8, "156": 8, "1582": 8, "0054843": 8, "50": 8, "5279": 8, "310": 8, "0745": 8, "91164684775759": 8, "35360u": 8, "93014bk": 8, "44021735": 8, "00019061": 8, "20292": 8, "35360": 8, "7343": 8, "127": 8, "2487": 8, "0071107": 8, "5913": 8, "9635": 8, "86997880798827": 8, "meteor": 8, "35364u": 8, "88005y": 8, "22067": 8, "81503681": 8, "00001147": 8, "84240": 8, "35364": 8, "82": 8, "5500": 8, "92": 8, "4124": 8, "0018834": 8, "303": 8, "2489": 8, "178": 8, "0638": 8, "94853833332534": 8, "data": [8, 9, 10], "store": 8, "nx2x3": 8, "6311e": 8, "7839e": 8, "4360e": 8, "0234e": 8, "2668e": 8, "7035e": 8, "1980e": 8, "6593e": 8, "4358e": 8, "1162e": 8, "3107e": 8, "1017e": 8, "2523e": 8, "4240e": 8, "3653e": 8, "8965e": 8, "5749e": 8, "2384e": 8, "1251e": 8, "4071e": 8, "5553e": 8, "9466e": 8, "4004e": 8, "7969e": 8, "2865e": 8, "4165e": 8, "0388e": 8, "9227e": 8, "7617e": 8, "7790e": 8, "6375e": 8, "4823e": 8, "7184e": 8, "5514e": 8, "6690e": 8, "tackl": [8, 10], "interest": 8, "multipl": 8, "omega": 8, "motion": [8, 9], "known": 8, "no_kozai": 8, "rad": [8, 9], "eccentr": [8, 9], "inclin": [8, 9], "right": [8, 9], "ascens": [8, 9], "ascend": [8, 9], "node": [8, 9], "argument": [8, 9], "perige": [8, 9], "anomali": [8, 9], "bstar": [8, 9], "earth": [8, 9], "radii": 8, "radian": 8, "bmatrix": 8, "frac": 8, "select": 8, "3958e": 8, "7748e": 8, "8397e": 8, "1363e": 8, "8542e": 8, "3558e": 8, "9710e": 8, "0059e": 8, "6121e": 8, "0092e": 8, "1681e": 8, "9048e": 8, "4880e": 8, "4406e": 8, "4768e": 8, "6585e": 8, "8450e": 8, "1588e": 8, "8879e": 8, "0836e": 8, "8834e": 8, "8445e": 8, "5107e": 8, "0606e": 8, "5196e": 8, "7975e": 8, "2453e": 8, "3748e": 8, "9690e": 8, "3317e": 8, "3379e": 8, "3524e": 8, "1467e": 8, "9327e": 8, "3608e": 8, "5900e": 8, "2770e": 8, "0481e": 8, "4477e": 8, "7844e": 8, "6570e": 8, "3067e": 8, "9218e": 8, "4145e": 8, "9367e": 8, "4110e": 8, "0053e": 8, "1622e": 8, "3502e": 8, "8547e": 8, "9912e": 8, "7381e": 8, "4493e": 8, "3289e": 8, "0005e": 8, "3278e": 8, "8571e": 8, "1566e": 8, "9009e": 8, "3436e": 8, "4896e": 8, "08": 8, "5904e": 8, "3934e": 8, "0740e": 8, "4019e": 8, "0689e": 8, "8966e": 8, "5064e": 8, "2860e": 8, "3674e": 8, "3401e": 8, "3902e": 8, "2270e": 8, "0189e": 8, "4565e": 8, "0912e": 8, "2672e": 8, "0517e": 8, "4685e": 8, "8340e": 8, "2430e": 8, "5945e": 8, "8755e": 8, "4366e": 8, "8901e": 8, "3780e": 8, "9886e": 8, "7959e": 8, "4959e": 8, "9473e": 8, "1336e": 8, "4652e": 8, "3936e": 8, "0383e": 8, "5212e": 8, "8531e": 8, "6571e": 8, "8964e": 8, "1689e": 8, "1308e": 8, "0275e": 8, "4614e": 8, "0693e": 8, "4701e": 8, "6338e": 8, "3967e": 8, "3256e": 8, "6863e": 8, "3501e": 8, "3394e": 8, "3717e": 8, "5451e": 8, "9702e": 8, "3591e": 8, "8135e": 8, "4581e": 8, "0505e": 8, "6180e": 8, "8171e": 8, "8098e": 8, "3454e": 8, "9161e": 8, "2823e": 8, "9310e": 8, "4067e": 8, "0037e": 8, "8593e": 8, "3677e": 8, "1533e": 8, "1424e": 8, "0376e": 8, "4503e": 8, "3371e": 8, "2630e": 8, "8570e": 8, "3159e": 8, "9008e": 8, "1989e": 8, "0113e": 8, "3975e": 8, "4020e": 8, "0739e": 8, "4105e": 8, "0561e": 8, "2093e": 8, "1133e": 8, "3658e": 8, "3884e": 8, "8926e": 8, "0129e": 8, "6303e": 8, "5565e": 8, "8336e": 8, "0333e": 8, "4081e": 8, "9066e": 8, "0564e": 8, "9215e": 8, "3996e": 8, "0006e": 8, "1097e": 8, "3973e": 8, "6503e": 8, "0468e": 8, "5362e": 8, "4527e": 8, "3506e": 8, "7577e": 8, "3932e": 8, "8566e": 8, "2164e": 8, "9003e": 8, "2600e": 8, "0747e": 8, "4162e": 8, "0734e": 8, "4248e": 8, "0347e": 8, "7296e": 8, "4525e": 8, "4216e": 8, "3626e": 8, "3400e": 8, "3850e": 8, "0001e": 8, "0029e": 8, "6223e": 8, "0019e": 8, "1995e": 8, "0516e": 8, "6971e": 8, "8318e": 8, "6264e": 8, "5180e": 8, "8888e": 8, "7857e": 8, "9036e": 8, "3870e": 8, "9942e": 8, "2124e": 8, "4532e": 8, "5661e": 8, "7141e": 8, "4547e": 8, "4589e": 8, "3754e": 8, "7026e": 8, "4673e": 8, "8550e": 8, "7691e": 8, "8985e": 8, "1711e": 8, "3420e": 8, "4750e": 8, "4423e": 8, "0715e": 8, "4509e": 8, "9417e": 8, "6882e": 8, "1468e": 8, "3559e": 8, "3398e": 8, "3779e": 8, "0392e": 8, "9843e": 8, "1991e": 8, "8951e": 8, "7616e": 8, "0511e": 8, "9173e": 8, "8250e": 8, "6002e": 8, "5175e": 8, "8889e": 8, "7810e": 8, "9037e": 8, "2011e": 8, "4529e": 8, "5612e": 8, "7052e": 8, "4498e": 8, "3753e": 8, "6862e": 8, "4669e": 8, "7555e": 8, "1457e": 8, "4782e": 8, "4421e": 8, "4508e": 8, "9439e": 8, "6831e": 8, "1316e": 8, "5741e": 8, "0284e": 8, "9844e": 8, "1907e": 8, "8957e": 8, "7566e": 8, "9122e": 8, "8251e": 8, "9558e": 8, "7134e": 8, "8527e": 8, "5262e": 8, "8672e": 8, "3635e": 8, "9778e": 8, "5696e": 8, "0120e": 8, "2693e": 8, "4787e": 8, "4238e": 8, "2248e": 8, "6098e": 8, "8484e": 8, "8180e": 8, "8915e": 8, "7561e": 8, "6169e": 8, "2715e": 8, "4931e": 8, "0642e": 8, "5019e": 8, "1030e": 8, "5820e": 8, "1322e": 8, "8689e": 8, "3389e": 8, "3386e": 8, "3599e": 8, "0061e": 8, "9463e": 8, "2881e": 8, "6719e": 8, "6241e": 8, "0492e": 8, "7910e": 8, "7981e": 8, "2738e": 8, "5104e": 8, "7222e": 8, "9049e": 8, "3879e": 8, "9947e": 8, "0593e": 8, "4491e": 8, "5000e": 8, "5939e": 8, "4584e": 8, "3736e": 8, "4819e": 8, "4620e": 8, "8552e": 8, "5849e": 8, "8987e": 8, "8276e": 8, "0503e": 8, "5184e": 8, "4404e": 8, "0717e": 8, "9713e": 8, "6190e": 8, "9427e": 8, "5639e": 8, "3564e": 8, "3784e": 8, "8922e": 8, "9856e": 8, "0857e": 8, "9029e": 8, "6937e": 8, "0512e": 8, "8489e": 8, "8257e": 8, "4736e": 8, "7849e": 8, "8375e": 8, "2403e": 8, "8519e": 8, "3545e": 8, "9698e": 8, "0327e": 8, "6194e": 8, "1171e": 8, "0130e": 8, "4897e": 8, "4434e": 8, "5209e": 8, "6668e": 8, "8443e": 8, "1889e": 8, "8872e": 8, "1393e": 8, "9284e": 8, "7715e": 8, "5137e": 8, "0599e": 8, "5225e": 8, "7449e": 8, "3582e": 8, "9859e": 8, "3304e": 8, "3378e": 8, "3511e": 8, "1706e": 8, "9304e": 8, "5430e": 8, "5759e": 8, "3882e": 8, "5596e": 8, "7818e": 8, "stai": 8, "activ": 8, "contain": [8, 9], "0152e": 8, "2822e": 8, "5574e": 8, "1865e": 8, "4201e": 8, "4554e": 8, "2169e": 8, "8311e": 8, "0519e": 8, "2915e": 8, "8070e": 8, "0682e": 8, "2995e": 8, "3906e": 8, "0370e": 8, "4344e": 8, "0217e": 8, "1228e": 8, "0224e": 8, "0858e": 8, "2339e": 8, "6793e": 8, "2243e": 8, "8699e": 8, "6678e": 8, "8580e": 8, "7899e": 8, "7475e": 8, "1514e": 8, "7465e": 8, "6525e": 8, "7371e": 8, "6045e": 8, "9503e": 8, "1032e": 8, "1148e": 8, "5543e": 8, "8858e": 8, "6449e": 8, "8946e": 8, "5038e": 8, "5616e": 8, "4836e": 8, "5470e": 8, "4668e": 8, "8317e": 8, "0259e": 8, "1128e": 8, "7921e": 8, "1149e": 8, "7412e": 8, "2118e": 8, "0312e": 8, "7194e": 8, "0365e": 8, "5328e": 8, "3791e": 8, "6447e": 8, "5367e": 8, "6423e": 8, "5391e": 8, "2733e": 8, "9985e": 8, "6292e": 8, "6476e": 8, "4235e": 8, "6494e": 8, "7612e": 8, "6006e": 8, "4560e": 8, "6430e": 8, "3995e": 8, "1429e": 8, "8838e": 8, "9900e": 8, "0215e": 8, "6400e": 8, "9150e": 8, "6609e": 8, "3064e": 8, "7965e": 8, "6847e": 8, "9099e": 8, "2441e": 8, "9595e": 8, "2284e": 8, "9755e": 8, "3447e": 8, "6382e": 8, "3686e": 8, "7157e": 8, "5790e": 8, "7950e": 8, "5267e": 8, "5597e": 8, "4834e": 8, "4756e": 8, "2361e": 8, "4863e": 8, "5500e": 8, "6660e": [8, 10], "6549e": 8, "9903e": 8, "4689e": 8, "3803e": 8, "5123e": 8, "1241e": 8, "7990e": 8, "4441e": 8, "1982e": 8, "0624e": 8, "8691e": 8, "1578e": 8, "7656e": 8, "9027e": 8, "0589e": 8, "0678e": 8, "1503e": 8, "8441e": 8, "5072e": 8, "7897e": 8, "6818e": 8, "4337e": 8, "6054e": 8, "8810e": 8, "6286e": 8, "8836e": 8, "0195e": 8, "0352e": 8, "7367e": 8, "7696e": 8, "7846e": 8, "8605e": 8, "5327e": 8, "7396e": 8, "9600e": 8, "9691e": 8, "9822e": 8, "9768e": 8, "7629e": 8, "7887e": 8, "1906e": 8, "0417e": 8, "7237e": 8, "0600e": 8, "5613e": 8, "3240e": 8, "2311e": 8, "8459e": 8, "2268e": 8, "8639e": 8, "2332e": 8, "7673e": 8, "7056e": 8, "8234e": 8, "0826e": 8, "4688e": 8, "0815e": 8, "4486e": 8, "4159e": 8, "3999e": 8, "6770e": 8, "4363e": 8, "7333e": 8, "2827e": 8, "3484e": 8, "8723e": 8, "7173e": 8, "0733e": 8, "8003e": 8, "0948e": 8, "6608e": 8, "2980e": 8, "6978e": 8, "8466e": 8, "7235e": 8, "3491e": 8, "5509e": 8, "1769e": 8, "5302e": 8, "9662e": 8, "4487e": 8, "3845e": 8, "1669e": 8, "9130e": 8, "6875e": 8, "0572e": 8, "8636e": 8, "8373e": 8, "7307e": 8, "4606e": 8, "0013e": 8, "5880e": 8, "1569e": 8, "5591e": 8, "1178e": 8, "8624e": 8, "4886e": 8, "2072e": 8, "1983e": 8, "5860e": 8, "1752e": 8, "1749e": 8, "4383e": 8, "8370e": 8, "9358e": 8, "8001e": 8, "0604e": 8, "7855e": 8, "7864e": 8, "4766e": 8, "5145e": 8, "8322e": 8, "2369e": 8, "2602e": 8, "4107e": 8, "3723e": 8, "9016e": 8, "6005e": 8, "3951e": 8, "6635e": 8, "2372e": 8, "9990e": 8, "2253e": 8, "1512e": 8, "usag": 9, "allow": 9, "user": 9, "string": 9, "class": 9, "easi": 9, "manipul": 9, "valu": 9, "well": 9, "interfac": 9, "here": [9, 10], "list": 9, "two": [9, 10], "plu": 9, "indic": 9, "satellit": 9, "name": 9, "tle_lin": 9, "timat": 9, "2847u": 9, "67053e": 9, "24063": 9, "46171465": 9, "00000366": 9, "27411": 9, "2847": 9, "69": 9, "9643": 9, "216": 9, "8651": 9, "0003597": 9, "77": 9, "7866": 9, "282": 9, "3646": 9, "02285835897007": 9, "access": 9, "info": 9, "dictionari": 9, "attribut": 9, "catalog": 9, "satellite_catalog_numb": [9, 10], "classif": 9, "intern": 9, "design": 9, "international_design": 9, "epoch": 9, "year": 9, "epoch_year": 9, "epoch_dai": 9, "_ndot": 9, "_nddot": 9, "drag": 9, "term": 9, "_bstar": 9, "element_numb": 9, "_inclo": 9, "_nodeo": 9, "_ecco": 9, "_argpo": 9, "_mo": 9, "_no_kozai": 9, "2024": 9, "63": 9, "1090112955380637e": 9, "00027411000000000004": 9, "999": 9, "2211073938530685": 9, "785010027666755": 9, "3576322839318213": 9, "928191961076781": 9, "061186262187069844": 9, "few": 9, "perhap": 9, "handi": 9, "radiu": 9, "accord": 9, "wsg": 9, "r_earth": 9, "util": 9, "get_gravity_const": 9, "wg": 9, "1e3": 9, "semi": 9, "major": 9, "axi": 9, "semi_major_axi": 9, "apoge": 9, "altitud": 9, "apogee_alt": 9, "perigee_alt": 9, "7264": 9, "027802311157": 9, "888": 9, "5036731116484": 9, "883": 9, "2779315106654": 9, "tle_dictionari": 9, "dict": 9, "43437": 9, "18100a": 9, "2020": 9, "143": 9, "90384230": 9, "ephemeris_typ": 9, "revolution_number_at_epoch": 9, "56353": 9, "mean_mot": 9, "0011": 9, "mean_motion_first_deriv": 9, "mean_motion_second_deriv": 9, "0221": 9, "7074": 9, "argument_of_perige": 9, "1627": 9, "raan": 9, "3618": 9, "mean_anomali": 9, "5224": 9, "b_star": 9, "0001": 9, "same": 9, "43437u": 9, "20143": 9, "00000000": 9, "10000": [9, 10], "99960": 9, "97": 9, "8268": 9, "249": 9, "9127": 9, "0221000": 9, "123": 9, "9136": 9, "259": 9, "1144": 9, "12608579563539": 9, "download": 9, "space": 9, "place": 9, "simpli": 9, "cannot": 9, "extra": 9, "descript": 9, "specifi": 9, "should": 9, "wai": 9, "34456u": 9, "93036sz": 9, "85349922": 9, "00005488": 9, "13805": 9, "34456": 9, "0462": 9, "226": 9, "5559": 9, "0056524": 9, "251": 9, "9784": 9, "107": 9, "5213": 9, "50477917685046": 9, "34457u": 9, "93036ta": 9, "86184197": 9, "00003162": 9, "55362": 9, "34457": 9, "0015": 9, "304": 9, "7929": 9, "0101048": 9, "52": 9, "3409": 9, "79": 9, "2774": 9, "64945713693281": 9, "inlin": 10, "would": 10, "onli": 10, "befor": 10, "small": 10, "perform": 10, "penalti": 10, "1812e": 10, "7596e": 10, "7231e": 10, "2926e": 10, "5809e": 10, "2651e": 10, "8993e": 10, "been": 10, "possibl": 10, "enough": 10, "pass": 10, "boolean": 10, "flag": 10, "default": 10, "plot_orbit": 10, "color": 10, "lightcor": 10, "satcat": 10, "axes3d": 10, "zlabel": 10, "parallel": 10, "need": 10, "prepar": 10, "mani": 10, "cat": 10, "sure": 10, "tles_batch": 10, "000": 10, "states_tem": 10, "ax": 10, "20000": 10, "darkkhaki": 10, "lightseagreen": 10, "These": 11, "includ": 11, "task": 11, "autodiff": 11, "complex": 11, "similar": 11, "covari": 11, "optim": 11}, "objects": {}, "objtypes": {}, "objnames": {}, "titleterms": {"api": 0, "capabl": 1, "credit": 2, "partial": [3, 8], "textrm": 3, "sgp4": 3, "document": 3, "get": [3, 4], "start": 3, "content": 3, "instal": 4, "depend": 4, "packag": 4, "conda": 4, "pip": 4, "from": [4, 6, 9], "sourc": 4, "verifi": 4, "help": 4, "covari": [5, 6], "propag": [5, 10], "similar": 6, "transform": 6, "cartesian": 6, "tle": [6, 8, 9, 10], "gradient": 7, "base": 7, "optim": 7, "problem": 7, "descript": 7, "deriv": 8, "comput": 8, "via": 8, "autodiff": 8, "respect": 8, "time": 8, "singl": [8, 10], "batch": [8, 10], "paramet": 8, "object": 9, "import": 9, "load": 9, "str": 9, "dic": 9, "file": 9, "tutori": 11, "basic": 11, "advanc": 11}, "envversion": {"sphinx.domains.c": 2, "sphinx.domains.changeset": 1, "sphinx.domains.citation": 1, "sphinx.domains.cpp": 8, "sphinx.domains.index": 1, "sphinx.domains.javascript": 2, "sphinx.domains.math": 2, "sphinx.domains.python": 3, "sphinx.domains.rst": 2, "sphinx.domains.std": 2, "sphinx.ext.intersphinx": 1, "sphinx": 57}, "alltitles": {"API": [[0, "api"]], "Capabilities": [[1, "capabilities"]], "Credits": [[2, "credits"]], "\\partial\\textrm{SGP4} Documentation": [[3, "partial-textrm-sgp4-documentation"]], "Getting Started": [[3, null]], "Contents": [[3, null]], "Installation": [[4, "installation"]], "Dependencies": [[4, "dependencies"]], "Packages": [[4, "packages"]], "conda": [[4, "conda"]], "pip": [[4, "pip"]], "Installation from source": [[4, "installation-from-source"]], "Verifying the installation": [[4, "verifying-the-installation"]], "Getting help": [[4, "getting-help"]], "Covariance Propagation": [[5, "covariance-propagation"]], "Similarity Transformation - from Cartesian to TLE Covariance": [[6, "similarity-transformation-from-cartesian-to-tle-covariance"]], "Covariance Transformation": [[6, "covariance-transformation"]], "Gradient Based Optimization": [[7, "gradient-based-optimization"]], "Problem description:": [[7, "problem-description"]], "Partial Derivatives Computation via Autodiff": [[8, "partial-derivatives-computation-via-autodiff"]], "Partials with respect to time": [[8, "partials-with-respect-to-time"]], "Single TLEs": [[8, "single-tles"], [8, "id1"]], "Batch TLEs": [[8, "batch-tles"]], "Partials with respect to TLE parameters": [[8, "partials-with-respect-to-tle-parameters"]], "Batch TLEs:": [[8, "id2"]], "TLE Object": [[9, "tle-object"]], "Imports": [[9, "imports"]], "Load TLE from str": [[9, "load-tle-from-str"]], "Load TLE from dic": [[9, "load-tle-from-dic"]], "Load TLEs from file:": [[9, "load-tles-from-file"]], "Propagate TLEs": [[10, "propagate-tles"]], "Single TLE propagation": [[10, "single-tle-propagation"]], "Batch TLE propagation": [[10, "batch-tle-propagation"]], "Tutorials": [[11, "tutorials"]], "Basics": [[11, "basics"]], "Advanced": [[11, "advanced"]]}, "indexentries": {}})