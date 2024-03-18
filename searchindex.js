Search.setIndex({"docnames": ["api", "capabilities", "credits", "index", "install", "notebooks/covariance_propagation", "notebooks/covariance_transformation", "notebooks/gradient_based_optimization", "notebooks/sgp4_partial_derivatives", "notebooks/tle_object", "notebooks/tle_propagation", "tutorials"], "filenames": ["api.rst", "capabilities.ipynb", "credits.ipynb", "index.md", "install.rst", "notebooks/covariance_propagation.ipynb", "notebooks/covariance_transformation.ipynb", "notebooks/gradient_based_optimization.ipynb", "notebooks/sgp4_partial_derivatives.ipynb", "notebooks/tle_object.ipynb", "notebooks/tle_propagation.ipynb", "tutorials.rst"], "titles": ["API", "Capabilities", "Credits", "<span class=\"math notranslate nohighlight\">\\(\\partial\\textrm{SGP4}\\)</span> Documentation", "Installation", "Covariance Propagation", "Similarity Transformation - from Cartesian to TLE Covariance", "Gradient Based Optimization", "<span class=\"math notranslate nohighlight\">\\(\\partial\\textrm{SGP4}\\)</span> Autodiff Support", "TLE Object", "Propagate TLEs with <span class=\"math notranslate nohighlight\">\\(\\partial\\textrm{SGP4}\\)</span>", "Tutorials"], "terms": {"partialtextrm": 0, "sgp4": [0, 1, 2, 6, 7, 11], "dsgp4": [1, 3, 4, 5, 6, 7, 8, 9, 10, 11], "i": [1, 3, 4, 5, 6, 7, 8, 10], "an": [1, 4, 5, 6, 7], "open": [1, 4], "sourc": 1, "project": [1, 2, 4], "constitut": 1, "differenti": [1, 3, 6, 7, 8], "version": 1, "partial": [2, 5, 6, 7, 11], "textrm": [2, 5, 6, 7, 11], "wa": [2, 4, 6, 9], "develop": [2, 3, 4], "dure": [2, 10], "sponsor": 2, "univers": 2, "oxford": 2, "while": 2, "giacomo": [2, 3], "acciarini": [2, 3], "ox4ailab": 2, "collabor": 2, "dr": 2, "at\u0131l\u0131m": [2, 3], "g\u00fcne\u015f": [2, 3], "baydin": [2, 3], "The": [2, 3, 8, 9], "main": [2, 3], "ar": [2, 3, 6, 7, 8, 10, 11], "gmail": [2, 3], "com": [2, 3, 4], "gune": 2, "robot": 2, "ox": 2, "ac": 2, "uk": 2, "program": 3, "written": [3, 8], "leverag": [3, 6, 11], "pytorch": [3, 8], "machin": 3, "learn": 3, "framework": [3, 11], "thi": [3, 4, 5, 6, 7, 8, 9, 10], "enabl": 3, "featur": [3, 8], "like": [3, 10], "automat": [3, 8], "batch": [3, 5], "propag": [3, 6, 7, 8, 9, 11], "across": 3, "differ": [3, 7], "tle": [3, 5, 7, 11], "were": 3, "previous": 3, "avail": [3, 4, 5, 6], "origin": [3, 8], "implement": [3, 8], "author": 3, "dario": 3, "izzo": 3, "instal": 3, "capabl": 3, "credit": 3, "tutori": [3, 6, 8, 9], "api": [3, 8, 9, 10], "ha": [4, 9, 10], "follow": [4, 7, 8, 9], "python": 4, "first": [4, 5, 6, 8, 9, 10], "we": [4, 5, 6, 7, 8, 9, 10], "add": 4, "forg": 4, "channel": 4, "config": 4, "set": [4, 9], "channel_prior": 4, "strict": 4, "now": [4, 5, 6, 7, 8, 9], "can": [4, 5, 6, 7, 8, 9, 10], "either": [4, 9], "through": 4, "mamba": 4, "pypi": 4, "http": 4, "org": 4, "you": 4, "via": [4, 7, 8], "us": [4, 5, 6, 8, 9, 10, 11], "git": 4, "clone": 4, "github": 4, "esa": 4, "usual": [4, 5], "pr": 4, "base": [4, 11], "workflow": 4, "thu": 4, "": [4, 5, 6, 8, 9, 10], "master": 4, "branch": 4, "normal": 4, "kept": 4, "work": [4, 8], "state": [4, 5, 6, 7, 8], "successfulli": 4, "compil": 4, "run": 4, "test": 4, "To": 4, "do": [4, 5, 6, 7], "so": [4, 7], "must": 4, "option": 4, "pytest": 4, "If": [4, 5, 6, 10], "command": 4, "execut": 4, "without": 4, "ani": [4, 7], "error": 4, "your": 4, "readi": 4, "troubl": 4, "pleas": 4, "hesit": 4, "contact": 4, "u": [4, 5, 6, 9, 10], "issu": 4, "report": 4, "In": [5, 6, 7, 8, 9, 10], "notebook": [5, 6, 8, 9, 10], "discuss": [5, 6, 9], "how": [5, 6, 8, 9, 10, 11], "appli": [5, 6, 9], "order": [5, 7], "approxim": [5, 7], "matrix": [5, 6, 8], "begin": [5, 6, 7, 8], "equat": [5, 6, 7, 8], "p_": [5, 6], "pmb": [5, 8], "x": [5, 7, 8, 10], "_f": 5, "dfrac": [5, 6, 7, 8], "_0": [5, 7], "_": [5, 7], "0": [5, 6, 7, 8, 9, 10], "t": [5, 6, 7, 8], "text": [5, 6], "end": [5, 6, 7, 8], "where": [5, 6, 7, 8], "initi": [5, 6, 7, 8, 10], "time": [5, 6, 7, 9, 10], "t_0": [5, 7], "vector": [5, 7, 8], "cartesian": [5, 7, 11], "teme": [5, 6], "certain": [5, 6], "t_f": 5, "import": [5, 6, 7, 8, 10], "aspect": 5, "notic": 5, "abov": [5, 6, 9], "formula": 5, "besid": 5, "also": [5, 8, 9, 10], "transform": [5, 11], "from": [5, 7, 11], "paramet": [5, 6], "ones": 5, "let": [5, 6, 8, 9, 10], "torch": [5, 6, 8, 10], "numpi": [5, 6, 9], "np": [5, 6], "And": 5, "construct": [5, 6, 8, 9, 10], "some": [5, 11], "object": [5, 8, 11], "inp_fil": [5, 8], "pslv": [5, 8], "deb": [5, 6, 7, 8, 9], "1": [5, 6, 7, 8, 9, 10], "35351u": [5, 8], "01049qk": [5, 8], "22066": [5, 8], "70636923": [5, 8], "00002156": [5, 8], "00000": [5, 6, 7, 8, 9], "63479": [5, 8], "3": [5, 6, 7, 8, 9, 10], "9999": [5, 6, 8], "2": [5, 6, 7, 8, 9, 10], "35351": [5, 8], "98": [5, 8], "8179": [5, 8], "29": [5, 8], "5651": [5, 8], "0005211": [5, 8], "45": [5, 8], "5944": [5, 8], "314": [5, 8], "5671": [5, 8], "14": [5, 6, 7, 8, 9], "44732274457505": [5, 8], "cosmo": [5, 6, 7, 8, 9], "2251": [5, 6, 7, 8, 9], "34550u": 5, "93036ul": 5, "22068": [5, 6, 7, 8, 9], "48847750": 5, "00005213": 5, "16483": 5, "9994": [5, 9], "34550": 5, "73": 5, "9920": 5, "106": 5, "8624": 5, "0177754": 5, "242": 5, "5189": 5, "115": 5, "7794": 5, "31752709674389": 5, "ggse": 5, "1292u": 5, "65016c": 5, "22069": 5, "19982496": 5, "00000050": 5, "68899": 5, "4": [5, 6, 7, 8, 9, 10], "9998": [5, 8], "1292": 5, "70": 5, "0769": 5, "62": 5, "6868": 5, "0022019": 5, "223": 5, "3241": 5, "136": 5, "6137": 5, "00692829906555": 5, "cz": 5, "2d": 5, "35275u": 5, "08056e": 5, "48794021": 5, "00000595": 5, "24349": 5, "35275": 5, "8424": 5, "84": [5, 9], "5826": 5, "0139050": 5, "155": 5, "2017": 5, "205": 5, "5930": 5, "25810852685662": 5, "fengyun": 5, "1c": 5, "35237u": 5, "99025dqv": 5, "17158141": 5, "00009246": 5, "14994": 5, "9992": [5, 8, 9], "35237": 5, "99": [5, 8], "5946": 5, "147": 5, "0483": 5, "0055559": 5, "41": 5, "0775": 5, "319": 5, "4589": 5, "70446744775205": 5, "splitlin": [5, 8], "tsinc": [5, 8, 10], "linspac": [5, 10], "5": [5, 6, 7, 8, 10], "60": [5, 10], "24": [5, 10], "8": [5, 6, 8], "dai": [5, 9, 10], "index": 5, "rang": [5, 6, 8], "len": [5, 8, 10], "k": [5, 7, 8], "append": [5, 6, 8, 9], "repeat": 5, "track": [5, 8, 9], "gradient": [5, 8, 11], "w": [5, 6, 7, 8], "tle_el": [5, 6, 8], "initialize_tl": [5, 6, 7, 8, 10], "with_grad": [5, 6, 8], "true": [5, 6, 8, 10], "orbit": [5, 10], "state_tem": [5, 6, 8, 10], "propagate_batch": [5, 8, 10], "output": [5, 8], "r": [5, 6, 7, 8], "build": [5, 8], "deriv": [5, 8, 9], "shape": [5, 8], "nx6x9": [5, 8], "n": [5, 8, 10], "number": [5, 8, 9], "element": [5, 6, 7, 8, 9, 10], "6": [5, 6, 8, 10], "9": [5, 6, 8, 10], "dx_dtle": [5, 6], "zero": [5, 6, 7, 8], "grad": [5, 6, 8], "none": [5, 6, 8], "flatten": [5, 6, 8], "backward": [5, 6, 8], "retain_graph": [5, 6, 8], "defin": [5, 7, 9], "cov_xyz": 5, "cov_rtn": 5, "cov_tl": [5, 6], "arrai": [5, 6], "06817079e": 5, "23": 5, "85804989e": 5, "25": 5, "51328946e": 5, "48167092e": 5, "13": [5, 6, 7, 8], "80784129e": 5, "11": [5, 9], "17516946e": 5, "17": [5, 8], "80719145e": 5, "47782854e": 5, "06374440e": 5, "19": 5, "10888880e": 5, "26": 5, "19571327e": 5, "28": 5, "57097512e": 5, "27": 5, "57618033e": 5, "16": 5, "87675528e": 5, "28440729e": 5, "20": [5, 7], "87608028e": 5, "57219640e": 5, "10539220e": 5, "22": [5, 7], "62208982e": 5, "00370060e": 5, "13145502e": 5, "41515501e": 5, "13025898e": 5, "12": [5, 7, 10], "11161104e": 5, "12805538e": 5, "40212641e": 5, "15": [5, 9], "61913778e": 5, "72347076e": 5, "25849762e": 5, "85837006e": 5, "69613552e": 5, "03": [5, 6, 8, 10], "60858400e": 5, "01": [5, 6, 8, 10], "44529381e": 5, "06": [5, 8], "60848714e": 5, "66633934e": 5, "04": [5, 6, 8, 10], "73554382e": 5, "10": [5, 6, 8], "98398791e": 5, "64526723e": 5, "81073778e": 5, "35783115e": 5, "70385711e": 5, "35662070e": 5, "60149046e": 5, "02": [5, 6, 8, 10], "68286334e": 5, "07": [5, 8], "00692291e": 5, "34937429e": 5, "18": [5, 8], "42650203e": 5, "75093676e": 5, "05": [5, 6, 8], "70379140e": 5, "42796071e": 5, "62336921e": 5, "98327475e": 5, "64467582e": 5, "80972744e": 5, "35541747e": 5, "60131157e": 5, "28248504e": 5, "71925407e": 5, "25500097e": 5, "85239630e": 5, "63783423e": 5, "21657063e": 5, "16764843e": 5, "73507662e": 5, "21": 5, "65971417e": 5, "73554368e": 5, "68286335e": 5, "28248505e": 5, "21656925e": 5, "21029182e": 5, "1000": [5, 7], "frob_norm": 5, "def": [5, 6], "rotation_matrix": [5, 6], "comput": [5, 6, 7, 8], "uvw": [5, 6], "rotat": [5, 6], "arg": [5, 6], "row": [5, 6], "column": [5, 6], "repres": [5, 6], "posit": [5, 6, 7, 8], "second": [5, 6, 8, 9], "veloc": [5, 6, 7, 8], "return": [5, 6, 7, 8], "v": [5, 6], "linalg": [5, 6], "norm": [5, 6], "cross": [5, 6], "vstack": [5, 6], "from_cartesian_to_rtn": [5, 6], "cartesian_to_rtn_rotation_matrix": [5, 6], "convert": [5, 6], "rtn": [5, 6], "frame": [5, 6], "suppli": [5, 6], "otherwis": [5, 6, 10], "r_rtn": [5, 6], "dot": [5, 6, 7, 8], "v_rtn": [5, 6], "stack": [5, 6], "final": [5, 8], "all": [5, 8, 9, 10], "idx": 5, "enumer": 5, "state_rtn": [5, 6], "detach": [5, 6], "6x6": 5, "transformation_matrix_cartesian_to_rtn": [5, 6], "matmul": [5, 6], "ord": 5, "fro": 5, "plot": [5, 10], "frobeniu": 5, "each": [5, 8, 10], "function": [5, 7], "matplotlib": [5, 10], "pyplot": 5, "plt": 5, "32": [5, 8], "40": 5, "semilogi": 5, "label": [5, 10], "line0": 5, "legend": 5, "xlabel": [5, 10], "min": [5, 9], "ylabel": [5, 10], "One": [6, 8], "obviou": 6, "applic": [6, 8], "its": [6, 7], "see": [6, 7, 8, 9], "covariance_propag": [6, 8], "ipynb": 6, "case": [6, 8, 10], "start": 6, "correspond": [6, 7], "given": [6, 7, 10], "refer": 6, "Then": [6, 9], "assum": [6, 9, 10], "have": [6, 7, 8, 9, 10], "associ": 6, "instanc": [6, 9], "fundament": 6, "astrodynam": 6, "vallado": 6, "theoret": 6, "background": 6, "want": [6, 7, 8, 10], "coordin": 6, "y": [6, 7, 8, 10], "m": [6, 8], "p_x": 6, "m_": 6, "ij": [6, 7], "y_i": [6, 7], "x_j": 6, "onc": 6, "done": [6, 7], "left": 6, "directli": [6, 9, 10], "which": [6, 7, 8, 10], "sever": 6, "exampl": [6, 7, 8, 9, 10, 11], "gener": [6, 7], "perturb": 6, "mean": [6, 7, 8, 9], "feed": 6, "them": [6, 8, 10], "noisi": 6, "observ": [6, 7], "As": [6, 8], "alwai": [6, 8, 10], "line": [6, 8, 9, 10], "34427u": 6, "93036ru": 6, "94647328": 6, "00008100": 6, "11455": 6, "34427": 6, "74": [6, 7, 8, 9], "0145": 6, "306": 6, "8269": 6, "0033346": 6, "0723": 6, "347": 6, "1308": 6, "76870515693886": 6, "my_tl": [6, 7, 10], "matrix_rtn": 6, "cov": 6, "cov_matrix_rtn": 6, "sigma_rr": 6, "sigma_rt": 6, "sigma_rn": 6, "sigma_vr": 6, "sigma_vt": 6, "sigma_vn": 6, "diag": 6, "covariance_diagon": 6, "100": [6, 7, 8], "1e": [6, 7, 9], "print": [6, 7, 8, 9], "f": [6, 7, 9, 10], "e": [6, 7, 8], "00": [6, 8, 10], "tensor": [6, 8, 10], "1941e": 6, "6009e": 6, "2616e": 6, "6578e": 6, "2582e": 6, "7": [6, 7, 8], "2697e": 6, "grad_fn": [6, 8], "transposebackward0": [6, 8], "c_teme": 6, "43678346e": 6, "27091412e": 6, "90611714e": 6, "00000000e": 6, "24494699e": 6, "42742252e": 6, "31826956e": 6, "80461804e": 6, "59800553e": 6, "09672885e": 6, "69441685e": 6, "57016478e": 6, "25009651e": 6, "quick": 6, "check": 6, "confirm": 6, "correct": 6, "c_rtn2": 6, "allclos": 6, "For": [6, 10], "detail": 6, "out": 6, "tle_propag": [6, 8], "6x9": 6, "screen": [6, 8], "0000e": [6, 8, 10], "7924e": 6, "5394e": 6, "0479e": 6, "5463e": 6, "3390e": [6, 8], "9523e": [6, 8], "1618e": 6, "3683e": [6, 8], "1736e": 6, "7875e": 6, "0267e": 6, "7277e": 6, "9559e": 6, "7716e": 6, "4301e": 6, "6362e": 6, "5385e": 6, "8152e": 6, "5521e": 6, "6044e": 6, "3649e": 6, "0581e": 6, "3574e": 6, "0789e": 6, "4300e": 6, "0851e": 6, "4891e": 6, "0766e": 6, "7732e": 6, "7622e": 6, "obtain": [6, 7], "9x9": 6, "pinv": 6, "thing": [6, 8], "correctli": [6, 7], "c_xyz_2": 6, "call": [7, 10], "look": 7, "futur": [7, 8, 10], "t_": 7, "ob": 7, "rightarrow": 7, "vec": 7, "z": [7, 8, 10], "find": 7, "That": 7, "when": 7, "abl": 7, "invert": 7, "formul": 7, "minimum": 7, "free": 7, "variabl": 7, "between": 7, "make": [7, 10], "current": 7, "reformul": 7, "align": 7, "minim": 7, "tild": 7, "newton": 7, "method": [7, 9, 10], "updat": [7, 9], "guess": 7, "y_": 7, "until": 7, "converg": 7, "df": 7, "y_k": 7, "jacobian": [7, 8], "respect": 7, "easili": 7, "made": [7, 8], "df_": 7, "j": [7, 8], "_1": 7, "_2": 7, "_3": 7, "_4": 7, "_5": 7, "_6": 7, "no_": 7, "kozai": 7, "ecco": 7, "inclo": 7, "mo": 7, "argpo": 7, "nodeo": 7, "n_": 7, "ddot": [7, 8], "b": [7, 8], "sinc": [7, 8], "built": 7, "input": 7, "quit": 7, "furthermor": 7, "found": 7, "simpl": [7, 11], "invers": 7, "keplerian": 7, "doe": [7, 10], "good": 7, "load": [7, 8, 10], "file_nam": 7, "extract": [7, 9, 10], "one": [7, 10], "expect": 7, "ident": [7, 8], "found_tl": 7, "newton_method": 7, "tle_0": 7, "time_mjd": 7, "date_mjd": 7, "new_tol": 7, "max_it": 7, "7582159128252637e": 7, "solut": 7, "iter": 7, "34454u": [7, 8, 9], "93036sx": [7, 8, 9], "91971155": [7, 8, 9], "00000319": [7, 8, 9], "11812": [7, 8, 9], "9996": [7, 8, 9], "34454": [7, 8, 9], "0583": [7, 8, 9], "280": [7, 8, 9], "7094": [7, 8, 9], "0037596": [7, 8, 9], "327": [7, 8, 9], "9100": [7, 8, 9], "31": [7, 8, 9], "9764": [7, 8, 9], "35844873683320": [7, 8, 9], "91971967": 7, "9997": [7, 9], "minut": [7, 8], "after": 7, "800": 7, "still": 7, "cours": 7, "547518096460396e": 7, "24138": 7, "254": 7, "2494": 7, "0037442": 7, "103": 7, "1744": 7, "5962": 7, "36399602683320": 7, "show": [8, 9, 10], "due": 8, "fact": 8, "autograd": 8, "more": [8, 11], "advanc": 8, "practic": 8, "state_transition_matrix_comput": 8, "graident_based_optim": 8, "orbit_determin": 8, "creat": [8, 9], "shown": 8, "howev": 8, "instead": 8, "standard": 8, "requir": [8, 10], "record": 8, "oper": 8, "variou": 8, "take": 8, "random": 8, "rand": 8, "requires_grad": 8, "fals": [8, 10], "4265e": 8, "9768e": 8, "6842e": 8, "9442e": 8, "6877e": 8, "1925e": 8, "3543e": 8, "9998e": 8, "0487e": 8, "0002e": 8, "8706e": 8, "2020e": 8, "4144e": 8, "9814e": 8, "2361e": 8, "9539e": 8, "2091e": 8, "1948e": 8, "3612e": 8, "9980e": 8, "2977e": 8, "9951e": 8, "1369e": 8, "2016e": 8, "3866e": 8, "9908e": 8, "2210e": 8, "9757e": 8, "1244e": 8, "1991e": 8, "3800e": 8, "9928e": 8, "9788e": 8, "9808e": 8, "8654e": 8, "1999e": 8, "4330e": 8, "9741e": 8, "9265e": 8, "9389e": 8, "9463e": 8, "1910e": 8, "4311e": 8, "9749e": 8, "8561e": 8, "9404e": 8, "8712e": 8, "1915e": 8, "3348e": 8, "0041e": 8, "4997e": 8, "0146e": 8, "1227e": 8, "2028e": 8, "3366e": 8, "0038e": 8, "1444e": 8, "0133e": 8, "1917e": 8, "2027e": 8, "retriev": 8, "v_x": 8, "v_y": 8, "v_z": 8, "type": 8, "d": 8, "dx": 8, "dt": 8, "dy": 8, "dz": 8, "2x": 8, "2y": 8, "2z": 8, "dv_x": 8, "dv_y": 8, "dv_z": 8, "care": 8, "about": 8, "mirror": 8, "km": [8, 9, 10], "henc": 8, "dimens": 8, "coher": 8, "si": 8, "convers": 8, "partial_deriv": 8, "zeros_lik": 8, "ones_lik": 8, "1665e": 8, "6127e": 8, "3155e": 8, "4217e": 8, "6080e": 8, "4397e": 8, "2001e": 8, "9224e": 8, "3212e": 8, "9460e": 8, "6239e": 8, "9458e": 8, "1724e": 8, "3255e": 8, "3169e": 8, "3418e": 8, "6112e": 8, "1430e": 8, "1970e": 8, "0822e": 8, "3210e": 8, "9915e": 8, "6227e": 8, "5943e": 8, "1854e": 8, "6747e": 8, "3195e": 8, "1593e": 8, "6177e": 8, "4709e": 8, "1885e": 8, "5193e": 8, "3199e": 8, "1154e": 8, "6191e": 8, "3105e": 8, "1633e": 8, "7678e": 8, "3146e": 8, "4648e": 8, "6062e": 8, "6001e": 8, "1643e": 8, "7227e": 8, "3149e": 8, "4523e": 8, "6067e": 8, "5534e": 8, "2088e": 8, "4736e": 8, "3217e": 8, "8176e": 8, "6269e": 8, "3182e": 8, "2080e": 8, "5150e": 8, "3216e": 8, "8295e": 8, "6266e": 8, "7451e": 8, "basic": 8, "35350u": 8, "01049qj": 8, "76869562": 8, "00000911": 8, "24939": 8, "35350": 8, "6033": 8, "64": 8, "7516": 8, "0074531": 8, "8340": 8, "261": 8, "1278": 8, "48029442457561": 8, "sl": 8, "35354u": 8, "93014bd": 8, "76520028": 8, "00021929": 8, "20751": 8, "9995": 8, "35354": 8, "75": 8, "7302": 8, "7819": 8, "0059525": 8, "350": 8, "7978": 8, "2117": 8, "92216400847487": 8, "35359u": 8, "93014bj": 8, "55187275": 8, "00025514": 8, "24908": 8, "35359": 8, "7369": 8, "156": 8, "1582": 8, "0054843": 8, "50": 8, "5279": 8, "310": 8, "0745": 8, "91164684775759": 8, "35360u": 8, "93014bk": 8, "44021735": 8, "00019061": 8, "20292": 8, "35360": 8, "7343": 8, "127": 8, "2487": 8, "0071107": 8, "5913": 8, "9635": 8, "86997880798827": 8, "meteor": 8, "35364u": 8, "88005y": 8, "22067": 8, "81503681": 8, "00001147": 8, "84240": 8, "35364": 8, "82": 8, "5500": 8, "92": 8, "4124": 8, "0018834": 8, "303": 8, "2489": 8, "178": 8, "0638": 8, "94853833332534": 8, "data": [8, 9, 10], "store": 8, "nx2x3": 8, "0735e": 8, "9531e": 8, "4324e": 8, "0411e": 8, "2566e": 8, "0200e": 8, "9395e": 8, "8043e": 8, "4349e": 8, "1177e": 8, "3061e": 8, "6737e": 8, "0685e": 8, "9491e": 8, "4202e": 8, "8165e": 8, "8836e": 8, "0019e": 8, "0046e": 8, "0486e": 8, "4125e": 8, "5379e": 8, "9955e": 8, "5682e": 8, "1282e": 8, "1445e": 8, "4114e": 8, "0593e": 8, "9011e": 8, "9192e": 8, "7865e": 8, "6324e": 8, "4897e": 8, "7086e": 8, "5589e": 8, "6639e": 8, "tackl": [8, 10], "interest": 8, "multipl": 8, "omega": 8, "motion": [8, 9], "known": 8, "no_kozai": 8, "rad": [8, 9], "eccentr": [8, 9], "inclin": [8, 9], "right": [8, 9], "ascens": [8, 9], "ascend": [8, 9], "node": [8, 9], "argument": [8, 9], "perige": [8, 9], "anomali": [8, 9], "bstar": [8, 9], "earth": [8, 9], "radii": 8, "radian": 8, "bmatrix": 8, "frac": 8, "select": 8, "0390e": 8, "5549e": 8, "8825e": 8, "0948e": 8, "8972e": 8, "3827e": 8, "9916e": 8, "9614e": 8, "4734e": 8, "8876e": 8, "2999e": 8, "7772e": 8, "4617e": 8, "3841e": 8, "7953e": 8, "4930e": 8, "8542e": 8, "6659e": 8, "8976e": 8, "8427e": 8, "7563e": 8, "2628e": 8, "4514e": 8, "0706e": 8, "4601e": 8, "7963e": 8, "0248e": 8, "1620e": 8, "6280e": 8, "3532e": 8, "3396e": 8, "3750e": 8, "7548e": 8, "9776e": 8, "7508e": 8, "8566e": 8, "0924e": 8, "0509e": 8, "2501e": 8, "8216e": 8, "8808e": 8, "4302e": 8, "9032e": 8, "0776e": 8, "9180e": 8, "3971e": 8, "9995e": 8, "5173e": 8, "4080e": 8, "8293e": 8, "3726e": 8, "7157e": 8, "4537e": 8, "3555e": 8, "3139e": 8, "4078e": 8, "8564e": 8, "7152e": 8, "9000e": 8, "3383e": 8, "0642e": 8, "9580e": 8, "4213e": 8, "0731e": 8, "4299e": 8, "0269e": 8, "9170e": 8, "9561e": 8, "4518e": 8, "3614e": 8, "3400e": 8, "3837e": 8, "3988e": 8, "9993e": 8, "9312e": 8, "9813e": 8, "0040e": 8, "0515e": 8, "1549e": 8, "8309e": 8, "7699e": 8, "6849e": 8, "8584e": 8, "2555e": 8, "8730e": 8, "3670e": 8, "9807e": 8, "8256e": 8, "5510e": 8, "0941e": 8, "4989e": 8, "9872e": 8, "4750e": 8, "4163e": 8, "1167e": 8, "5880e": 8, "8497e": 8, "0326e": 8, "8929e": 8, "6104e": 8, "4973e": 8, "4852e": 8, "0657e": 8, "4940e": 8, "2367e": 8, "2876e": 8, "0289e": 8, "8240e": 8, "3388e": 8, "3630e": 8, "4366e": 8, "8103e": 8, "7077e": 8, "3344e": 8, "0496e": 8, "4996e": 8, "8035e": 8, "2512e": 8, "4139e": 8, "9057e": 8, "5033e": 8, "9206e": 8, "3990e": 8, "0003e": 8, "2154e": 8, "4000e": 8, "6969e": 8, "1315e": 8, "5828e": 8, "4529e": 8, "3519e": 8, "9013e": 8, "3970e": 8, "3460e": 8, "9002e": 8, "6488e": 8, "4693e": 8, "0444e": 8, "4175e": 8, "0733e": 8, "4261e": 8, "0327e": 8, "7783e": 8, "5823e": 8, "4294e": 8, "3623e": 8, "3847e": 8, "1037e": 8, "0020e": 8, "7026e": 8, "9965e": 8, "6777e": 8, "0516e": 8, "0178e": 8, "8316e": 8, "6625e": 8, "2969e": 8, "9232e": 8, "9934e": 8, "9382e": 8, "4121e": 8, "0057e": 8, "4981e": 8, "3459e": 8, "7802e": 8, "6364e": 8, "6633e": 8, "4491e": 8, "3268e": 8, "8892e": 8, "8571e": 8, "0764e": 8, "9009e": 8, "1220e": 8, "0790e": 8, "08": 8, "6385e": 8, "3913e": 8, "0740e": 8, "3998e": 8, "0720e": 8, "8187e": 8, "6535e": 8, "2732e": 8, "3678e": 8, "3401e": 8, "3907e": 8, "0609e": 8, "0204e": 8, "1648e": 8, "0994e": 8, "4253e": 8, "0517e": 8, "9841e": 8, "8340e": 8, "8117e": 8, "4284e": 8, "9034e": 8, "0638e": 8, "9183e": 8, "3973e": 8, "9996e": 8, "4845e": 8, "4072e": 8, "8149e": 8, "3464e": 8, "7013e": 8, "4536e": 8, "3551e": 8, "2688e": 8, "4066e": 8, "6751e": 8, "9001e": 8, "2634e": 8, "9674e": 8, "4209e": 8, "4295e": 8, "0275e": 8, "9019e": 8, "9152e": 8, "4493e": 8, "3615e": 8, "3838e": 8, "3668e": 8, "9063e": 8, "9830e": 8, "8918e": 8, "1400e": 8, "8310e": 8, "3746e": 8, "3903e": 8, "9094e": 8, "6839e": 8, "9242e": 8, "4016e": 8, "0015e": 8, "7854e": 8, "3887e": 8, "5075e": 8, "7868e": 8, "3929e": 8, "4519e": 8, "3467e": 8, "3206e": 8, "3815e": 8, "8568e": 8, "8183e": 8, "9005e": 8, "6631e": 8, "6160e": 8, "1677e": 8, "0736e": 8, "4207e": 8, "0409e": 8, "5800e": 8, "0586e": 8, "3974e": 8, "3636e": 8, "3860e": 8, "6818e": 8, "0058e": 8, "3755e": 8, "0181e": 8, "7309e": 8, "2193e": 8, "8325e": 8, "7170e": 8, "4980e": 8, "8922e": 8, "6207e": 8, "9070e": 8, "3893e": 8, "8150e": 8, "4426e": 8, "3945e": 8, "2825e": 8, "4575e": 8, "3708e": 8, "1319e": 8, "4535e": 8, "8554e": 8, "2906e": 8, "8989e": 8, "2786e": 8, "5831e": 8, "5879e": 8, "4374e": 8, "0719e": 8, "4461e": 8, "5085e": 8, "6196e": 8, "5464e": 8, "3573e": 8, "3398e": 8, "3793e": 8, "6573e": 8, "9878e": 8, "9043e": 8, "9154e": 8, "5851e": 8, "0512e": 8, "7397e": 8, "8267e": 8, "3179e": 8, "5328e": 8, "8863e": 8, "9085e": 8, "3853e": 8, "9932e": 8, "5093e": 8, "4612e": 8, "6938e": 8, "9468e": 8, "4600e": 8, "3789e": 8, "1331e": 8, "4775e": 8, "8547e": 8, "1254e": 8, "8982e": 8, "8354e": 8, "9051e": 8, "3908e": 8, "4459e": 8, "0711e": 8, "4546e": 8, "8841e": 8, "8219e": 8, "5459e": 8, "5960e": 8, "3549e": 8, "3397e": 8, "3768e": 8, "3236e": 8, "9816e": 8, "4184e": 8, "8798e": 8, "8931e": 8, "0510e": 8, "0495e": 8, "8237e": 8, "5909e": 8, "6559e": 8, "8640e": 8, "9865e": 8, "8786e": 8, "3706e": 8, "9834e": 8, "1540e": 8, "5327e": 8, "8146e": 8, "9892e": 8, "7068e": 8, "4715e": 8, "4089e": 8, "0116e": 8, "5662e": 8, "8509e": 8, "2522e": 8, "8941e": 8, "4655e": 8, "3777e": 8, "6470e": 8, "4774e": 8, "0670e": 8, "4862e": 8, "9950e": 8, "2882e": 8, "7791e": 8, "3447e": 8, "3660e": 8, "8157e": 8, "9582e": 8, "3346e": 8, "7428e": 8, "0465e": 8, "0499e": 8, "2100e": 8, "8084e": 8, "stai": 8, "activ": 8, "contain": [8, 9], "9406e": 8, "2526e": 8, "2483e": 8, "8238e": 8, "0063e": 8, "1890e": 8, "4124e": 8, "8866e": 8, "2817e": 8, "4747e": 8, "3527e": 8, "9351e": 8, "9616e": 8, "3912e": 8, "0337e": 8, "0189e": 8, "4932e": 8, "3284e": 8, "1120e": 8, "2429e": 8, "6767e": 8, "2337e": 8, "1795e": 8, "3451e": 8, "1470e": 8, "7929e": 8, "7401e": 8, "1502e": 8, "7402e": 8, "0787e": 8, "4127e": 8, "6942e": 8, "4663e": 8, "8479e": 8, "1144e": 8, "2981e": 8, "8751e": 8, "8386e": 8, "8844e": 8, "3605e": 8, "6414e": 8, "3403e": 8, "5477e": 8, "4642e": 8, "0193e": 8, "0223e": 8, "1208e": 8, "9329e": 8, "1229e": 8, "7431e": 8, "2124e": 8, "0325e": 8, "7974e": 8, "0304e": 8, "9698e": 8, "0357e": 8, "6179e": 8, "7596e": [8, 10], "6235e": 8, "5373e": 8, "6419e": 8, "5398e": 8, "5922e": 8, "1778e": 8, "2827e": 8, "6132e": 8, "6449e": 8, "4228e": 8, "6467e": 8, "9369e": 8, "4501e": 8, "7358e": 8, "6651e": 8, "5703e": 8, "1428e": 8, "5455e": 8, "8809e": 8, "3300e": 8, "3986e": 8, "6756e": 8, "4001e": 8, "6967e": 8, "3283e": 8, "8080e": 8, "6192e": 8, "8659e": 8, "4110e": 8, "7393e": 8, "3845e": 8, "9655e": 8, "3001e": 8, "3215e": 8, "1909e": 8, "7183e": 8, "9018e": 8, "7978e": 8, "2470e": 8, "5850e": 8, "5777e": 8, "4267e": 8, "2392e": 8, "4370e": 8, "3571e": 8, "6662e": 8, "4598e": 8, "5986e": 8, "4810e": 8, "3807e": 8, "5250e": 8, "0294e": 8, "8379e": 8, "0299e": 8, "2745e": 8, "2234e": 8, "8697e": 8, "5427e": 8, "7762e": 8, "9527e": 8, "8276e": 8, "1640e": 8, "5506e": 8, "9304e": 8, "5124e": 8, "7134e": 8, "8560e": 8, "8645e": 8, "7304e": 8, "5863e": 8, "7544e": 8, "9292e": 8, "3850e": 8, "7484e": 8, "0140e": 8, "7247e": 8, "5275e": 8, "7737e": 8, "8482e": 8, "3658e": 8, "3542e": 8, "9837e": 8, "9641e": 8, "0087e": 8, "5110e": 8, "8992e": 8, "0938e": 8, "8570e": 8, "9595e": 8, "7131e": 8, "9783e": 8, "2313e": 8, "1662e": 8, "4814e": 8, "3312e": 8, "5701e": 8, "8611e": 8, "2921e": 8, "7258e": 8, "8473e": 8, "1163e": 8, "0491e": 8, "1153e": 8, "2941e": 8, "4596e": 8, "8795e": 8, "4660e": 8, "3577e": 8, "3931e": 8, "7237e": 8, "2742e": 8, "5756e": 8, "9202e": 8, "7207e": 8, "3948e": 8, "8036e": 8, "8266e": 8, "8397e": 8, "6883e": 8, "8494e": 8, "7136e": 8, "1105e": 8, "5042e": 8, "7577e": 8, "6237e": 8, "9779e": 8, "4508e": 8, "0207e": 8, "2896e": 8, "2035e": 8, "5565e": 8, "6377e": 8, "4808e": 8, "2534e": 8, "7433e": 8, "2810e": 8, "2558e": 8, "7989e": 8, "0270e": 8, "7697e": 8, "2653e": 8, "0669e": 8, "0008e": 8, "0660e": 8, "5317e": 8, "0431e": 8, "0676e": 8, "1818e": 8, "2923e": 8, "0636e": 8, "8908e": 8, "9890e": 8, "8649e": 8, "2638e": 8, "9084e": 8, "2554e": 8, "0379e": 8, "2437e": 8, "6934e": 8, "1263e": 8, "7706e": 8, "1096e": 8, "1090e": 8, "6885e": 8, "1012e": 8, "0949e": 8, "8701e": 8, "1575e": 8, "1057e": 8, "2665e": 8, "0942e": 8, "5982e": 8, "usag": 9, "allow": 9, "user": 9, "string": 9, "class": 9, "easi": 9, "manipul": 9, "valu": 9, "well": 9, "interfac": 9, "here": [9, 10], "list": 9, "two": [9, 10], "plu": 9, "indic": 9, "satellit": 9, "name": 9, "tle_lin": 9, "timat": 9, "2847u": 9, "67053e": 9, "24063": 9, "46171465": 9, "00000366": 9, "27411": 9, "2847": 9, "69": 9, "9643": 9, "216": 9, "8651": 9, "0003597": 9, "77": 9, "7866": 9, "282": 9, "3646": 9, "02285835897007": 9, "access": 9, "info": 9, "dictionari": 9, "attribut": 9, "catalog": 9, "satellite_catalog_numb": [9, 10], "classif": 9, "intern": 9, "design": 9, "international_design": 9, "epoch": 9, "year": 9, "epoch_year": 9, "epoch_dai": 9, "_ndot": 9, "_nddot": 9, "drag": 9, "term": 9, "_bstar": 9, "element_numb": 9, "_inclo": 9, "_nodeo": 9, "_ecco": 9, "_argpo": 9, "_mo": 9, "_no_kozai": 9, "2024": 9, "63": 9, "1090112955380637e": 9, "00027411000000000004": 9, "999": 9, "2211073938530685": 9, "785010027666755": 9, "3576322839318213": 9, "928191961076781": 9, "061186262187069844": 9, "few": 9, "perhap": 9, "handi": 9, "radiu": 9, "accord": 9, "wsg": 9, "r_earth": 9, "util": 9, "get_gravity_const": 9, "wg": 9, "1e3": 9, "semi": 9, "major": 9, "axi": 9, "semi_major_axi": 9, "apoge": 9, "altitud": 9, "apogee_alt": 9, "perigee_alt": 9, "7264": 9, "027802311157": 9, "888": 9, "5036731116484": 9, "883": 9, "2779315106654": 9, "tle_dictionari": 9, "dict": 9, "43437": 9, "18100a": 9, "2020": 9, "143": 9, "90384230": 9, "ephemeris_typ": 9, "revolution_number_at_epoch": 9, "56353": 9, "mean_mot": 9, "0011": 9, "mean_motion_first_deriv": 9, "mean_motion_second_deriv": 9, "0221": 9, "7074": 9, "argument_of_perige": 9, "1627": 9, "raan": 9, "3618": 9, "mean_anomali": 9, "5224": 9, "b_star": 9, "0001": 9, "same": 9, "43437u": 9, "20143": 9, "00000000": 9, "10000": [9, 10], "99960": 9, "97": 9, "8268": 9, "249": 9, "9127": 9, "0221000": 9, "123": 9, "9136": 9, "259": 9, "1144": 9, "12608579563539": 9, "download": 9, "space": 9, "place": 9, "simpli": 9, "cannot": 9, "extra": 9, "descript": 9, "specifi": 9, "should": 9, "wai": 9, "34456u": 9, "93036sz": 9, "85349922": 9, "00005488": 9, "13805": 9, "34456": 9, "0462": 9, "226": 9, "5559": 9, "0056524": 9, "251": 9, "9784": 9, "107": 9, "5213": 9, "50477917685046": 9, "34457u": 9, "93036ta": 9, "86184197": 9, "00003162": 9, "55362": 9, "34457": 9, "0015": 9, "304": 9, "7929": 9, "0101048": 9, "52": 9, "3409": 9, "79": 9, "2774": 9, "64945713693281": 9, "inlin": 10, "would": 10, "onli": 10, "befor": 10, "small": 10, "perform": 10, "penalti": 10, "1812e": 10, "6660e": 10, "7231e": 10, "2926e": 10, "5809e": 10, "2651e": 10, "8993e": 10, "been": 10, "possibl": 10, "enough": 10, "pass": 10, "boolean": 10, "flag": 10, "default": 10, "plot_orbit": 10, "color": 10, "lightcor": 10, "satcat": 10, "axes3d": 10, "zlabel": 10, "parallel": 10, "need": 10, "prepar": 10, "mani": 10, "cat": 10, "sure": 10, "tles_batch": 10, "000": 10, "states_tem": 10, "ax": 10, "20000": 10, "darkkhaki": 10, "lightseagreen": 10, "These": 11, "includ": 11, "task": 11, "autodiff": 11, "support": 11, "complex": 11, "similar": 11, "covari": 11, "optim": 11}, "objects": {}, "objtypes": {}, "objnames": {}, "titleterms": {"api": 0, "capabl": 1, "credit": 2, "partial": [3, 8, 10], "textrm": [3, 8, 10], "sgp4": [3, 8, 10], "document": 3, "get": [3, 4], "start": 3, "content": 3, "instal": 4, "depend": 4, "packag": 4, "conda": 4, "pip": 4, "from": [4, 6, 9], "sourc": 4, "verifi": 4, "help": 4, "covari": [5, 6], "propag": [5, 10], "similar": 6, "transform": 6, "cartesian": 6, "tle": [6, 8, 9, 10], "gradient": 7, "base": 7, "optim": 7, "problem": 7, "descript": 7, "autodiff": 8, "support": 8, "respect": 8, "time": 8, "singl": [8, 10], "batch": [8, 10], "paramet": 8, "object": 9, "import": 9, "load": 9, "str": 9, "dic": 9, "file": 9, "tutori": 11, "basic": 11, "advanc": 11}, "envversion": {"sphinx.domains.c": 2, "sphinx.domains.changeset": 1, "sphinx.domains.citation": 1, "sphinx.domains.cpp": 8, "sphinx.domains.index": 1, "sphinx.domains.javascript": 2, "sphinx.domains.math": 2, "sphinx.domains.python": 3, "sphinx.domains.rst": 2, "sphinx.domains.std": 2, "sphinx.ext.intersphinx": 1, "sphinx": 57}, "alltitles": {"API": [[0, "api"]], "Capabilities": [[1, "capabilities"]], "Credits": [[2, "credits"]], "\\partial\\textrm{SGP4} Documentation": [[3, "partial-textrm-sgp4-documentation"]], "Getting Started": [[3, null]], "Contents": [[3, null]], "Installation": [[4, "installation"]], "Dependencies": [[4, "dependencies"]], "Packages": [[4, "packages"]], "conda": [[4, "conda"]], "pip": [[4, "pip"]], "Installation from source": [[4, "installation-from-source"]], "Verifying the installation": [[4, "verifying-the-installation"]], "Getting help": [[4, "getting-help"]], "Covariance Propagation": [[5, "covariance-propagation"]], "Similarity Transformation - from Cartesian to TLE Covariance": [[6, "similarity-transformation-from-cartesian-to-tle-covariance"]], "Covariance Transformation": [[6, "covariance-transformation"]], "Gradient Based Optimization": [[7, "gradient-based-optimization"]], "Problem description:": [[7, "problem-description"]], "\\partial\\textrm{SGP4} Autodiff Support": [[8, "partial-textrm-sgp4-autodiff-support"]], "Partials with respect to time": [[8, "partials-with-respect-to-time"]], "Single TLEs": [[8, "single-tles"], [8, "id1"]], "Batch TLEs": [[8, "batch-tles"]], "Partials with respect to TLE parameters": [[8, "partials-with-respect-to-tle-parameters"]], "Batch TLEs:": [[8, "id2"]], "TLE Object": [[9, "tle-object"]], "Imports": [[9, "imports"]], "Load TLE from str": [[9, "load-tle-from-str"]], "Load TLE from dic": [[9, "load-tle-from-dic"]], "Load TLEs from file:": [[9, "load-tles-from-file"]], "Propagate TLEs with \\partial\\textrm{SGP4}": [[10, "propagate-tles-with-partial-textrm-sgp4"]], "Single TLE propagation": [[10, "single-tle-propagation"]], "Batch TLE propagation": [[10, "batch-tle-propagation"]], "Tutorials": [[11, "tutorials"]], "Basics": [[11, "basics"]], "Advanced": [[11, "advanced"]]}, "indexentries": {}})