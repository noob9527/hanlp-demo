import hanlp
# Customize HANLP_HOME
# All resources HanLP use will be cached into a directory called HANLP_HOME.
# It is an environment variable which you can customize to any path you like.
# By default, HANLP_HOME resolves to ~/.hanlp and %appdata%\hanlp on *nix and Windows respectively.
# If you want to redirect HANLP_HOME to a different location, say /data/hanlp,
# the following shell command can be very helpful.


# Pre-download all required models
hanlp.load(hanlp.pretrained.tok.FINE_ELECTRA_SMALL_ZH)
hanlp.load(hanlp.pretrained.ner.MSRA_NER_ELECTRA_SMALL_ZH)
hanlp.load(hanlp.pretrained.pos.CTB9_POS_ELECTRA_SMALL)
hanlp.load(hanlp.pretrained.pos.PKU_POS_ELECTRA_SMALL)


