from modules.geofolki.folki import GEFolkiIter, EFolkiIter, FolkiIter
from modules.geofolki.pyramid import BurtOF

GEFolki = BurtOF(GEFolkiIter)
EFolki = BurtOF(EFolkiIter)
Folki = BurtOF(FolkiIter)
