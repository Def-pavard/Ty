# Modifié main.py
ImporterImporter OS.
ImporterImporter  des systèmes.
# NE CHANGE PAS ÇA !!!
Syssys..path.insert (0, os.path.dirname (os.path.dirname (__file__)))path.insert (0, os.path.dirname (os.path.dirname (__file__)))
Importer logging.
DemandesDemandesDemandes d'importation.importation.'importation.importation.'importation.importation.'importation.importation.
à  partir de dotenv  import load_dotenv.partir de dotenv import load_dotenv.import load_dotenv.partir de dotenv import load_dotenv.
à  
 . flask_cors.   
    à   partir de src. routes.verifier import verifier_bppartir. de src.routes.verifier import verifier_bproutes.verifier import verifier_bppartir. de src.routes.verifier import verifier_bp routes.verifier import verifier_bppartir. de src.routes.verifier import verifier_bproutes.verifier import verifier_bppartir. de src.routes.verifier import verifier_bp

. = Flask (__name__, static_folder=os.path.join (os.path.dirname (__file__), 'static'))Flask (__name__, static_folder=os.path.join (os.path.dirname (__file__), 'static'))
..config..['secret_key' = ''ASDF..#FGSgvasgf$5$WGT'config['secret_key' = ''asdf#FGSgvasgf$5$WGT'config..['secret_key' = ''ASDF..#FGSgvasgf$5$WGT'config['secret_key' = ''asdf#FGSgvasgf$5$WGT'

# Activer CORS Pour ? ? toutes les.. routes
CORS (app, origins="*")"*")

Appapp..register_blueprint (verifier_bp, url_prefix=(verifier_bp, url_prefix='/.'/.api''

# Supprimer la configuration de SQLAlchemy car elle n'est pas utilisée dans le plan du vérificateurSupprimer
# Si nécessaire pour d'autres plans., rajouter avec les importations approprié.esautres plans., rajouter avec les importations approprié.es

@app.Route.. ('/', Défauts.éfauts={'chemin.':)App..Route. ('/', defaults={'path':)
@app.Route.. ('/<path :path>') app.route ('/<path :path>')
Def Servir.. (chemin) :
 static_folder_path = 
   si static_folder_path. est Aucun :    
  renvoyer "Dossier statique non configuré", 404   

   if chemin. !="" et OS..chemin..existe.. (OS..chemin..Rejoignez-nous !-nous ! (static_folder_path, path)) :    
 return send_from_directory (static_folder_path, path) . 
 Autre : 
  index_path = OS..chemin..Rejoignez-nous !-nous ! (static_folder_path, 'Index...html') os.path.join (static_folder_path, 'index.html')   
 si os.path.exists (index_path) : 
 return send_from_directory (static_folder_path, 'index.html') 
 Autre : 
  retour "index.html non trouvé", 404   

SISI __name__ == '__main__':
 app.run (host='0.0.0.0', port=5002, debug=True 
