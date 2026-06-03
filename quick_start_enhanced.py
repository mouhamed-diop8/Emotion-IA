#!/usr/bin/env python3
"""
Script de Démarrage Rapide - Système Amélioré
Vérifie l'installation et lance le système approprié
"""

import os
import sys
import subprocess

def print_header(text):
    """Afficher un en-tête formaté"""
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70)

def check_file_exists(filepath):
    """Vérifier si un fichier existe"""
    return os.path.exists(filepath)

def check_dependencies():
    """Vérifier les dépendances Python"""
    print_header("🔍 VÉRIFICATION DES DÉPENDANCES")
    
    dependencies = {
        'cv2': 'opencv-python',
        'tensorflow': 'tensorflow',
        'numpy': 'numpy',
        'pandas': 'pandas',
        'matplotlib': 'matplotlib',
        'seaborn': 'seaborn',
        'sklearn': 'scikit-learn'
    }
    
    missing = []
    
    for module, package in dependencies.items():
        try:
            __import__(module)
            print(f"✅ {package:20s} - Installé")
        except ImportError:
            print(f"❌ {package:20s} - MANQUANT")
            missing.append(package)
    
    if missing:
        print(f"\n⚠️  Dépendances manquantes: {', '.join(missing)}")
        print("\n💡 Pour installer:")
        print(f"   pip3 install {' '.join(missing)}")
        return False
    
    print("\n✅ Toutes les dépendances sont installées!")
    return True

def check_data():
    """Vérifier les données d'entraînement"""
    print_header("📂 VÉRIFICATION DES DONNÉES")
    
    train_path = 'train'
    test_path = 'test'
    
    if not os.path.exists(train_path):
        print(f"❌ Dossier d'entraînement manquant: {train_path}")
        return False
    
    if not os.path.exists(test_path):
        print(f"❌ Dossier de test manquant: {test_path}")
        return False
    
    # Compter les images
    train_count = sum([len(files) for r, d, files in os.walk(train_path)])
    test_count = sum([len(files) for r, d, files in os.walk(test_path)])
    
    print(f"✅ Dossier train: {train_count} images")
    print(f"✅ Dossier test: {test_count} images")
    
    if train_count < 100:
        print("⚠️  Peu d'images d'entraînement détectées")
        return False
    
    return True

def check_models():
    """Vérifier les modèles disponibles"""
    print_header("🤖 VÉRIFICATION DES MODÈLES")
    
    models = {
        'best_model_optimized.h5': 'Modèle optimisé (recommandé)',
        'best_model.h5': 'Modèle standard',
        'my_cnn_model.h5': 'Modèle CNN de base'
    }
    
    available_models = []
    
    for model_file, description in models.items():
        if check_file_exists(model_file):
            print(f"✅ {model_file:30s} - {description}")
            available_models.append(model_file)
        else:
            print(f"❌ {model_file:30s} - Non trouvé")
    
    return available_models

def main_menu():
    """Menu principal"""
    print_header("🎭 SYSTÈME DE RECONNAISSANCE D'ÉMOTIONS AMÉLIORÉ")
    
    print("\n📋 Options disponibles:\n")
    print("  1. 🏋️  Entraîner le modèle optimisé (recommandé)")
    print("  2. 🎥 Lancer la détection en temps réel (système amélioré)")
    print("  3. 🎥 Lancer la détection en temps réel (système standard)")
    print("  4. 📊 Vérifier l'installation")
    print("  5. 📚 Afficher le guide")
    print("  6. ❌ Quitter")
    
    choice = input("\n👉 Votre choix [1-6]: ").strip()
    return choice

def train_optimized_model():
    """Entraîner le modèle optimisé"""
    print_header("🏋️  ENTRAÎNEMENT DU MODÈLE OPTIMISÉ")
    
    if not check_data():
        print("\n❌ Impossible de continuer sans données d'entraînement")
        return
    
    print("\n⚠️  L'entraînement peut prendre 30-60 minutes")
    confirm = input("Continuer? [o/N]: ").strip().lower()
    
    if confirm == 'o':
        print("\n🚀 Démarrage de l'entraînement...")
        try:
            subprocess.run(['python3', 'train_optimized_cnn.py'])
        except KeyboardInterrupt:
            print("\n⚠️  Entraînement interrompu")
        except Exception as e:
            print(f"\n❌ Erreur: {e}")
    else:
        print("❌ Entraînement annulé")

def run_enhanced_detection():
    """Lancer la détection améliorée"""
    print_header("🎥 DÉTECTION EN TEMPS RÉEL - SYSTÈME AMÉLIORÉ")
    
    # Vérifier le modèle
    if check_file_exists('best_model_optimized.h5'):
        model_file = 'best_model_optimized.h5'
        print(f"✅ Utilisation du modèle: {model_file}")
    elif check_file_exists('best_model.h5'):
        model_file = 'best_model.h5'
        print(f"⚠️  Utilisation du modèle standard: {model_file}")
        print("💡 Entraînez le modèle optimisé pour de meilleures performances")
    else:
        print("❌ Aucun modèle trouvé!")
        print("💡 Entraînez d'abord un modèle (option 1)")
        return
    
    print("\n🚀 Démarrage de la détection...")
    print("📹 Assurez-vous que votre webcam est disponible")
    
    input("\nAppuyez sur Entrée pour continuer...")
    
    try:
        subprocess.run(['python3', 'webcam_enhanced_detection.py'])
    except KeyboardInterrupt:
        print("\n⚠️  Détection interrompue")
    except Exception as e:
        print(f"\n❌ Erreur: {e}")

def run_standard_detection():
    """Lancer la détection standard"""
    print_header("🎥 DÉTECTION EN TEMPS RÉEL - SYSTÈME STANDARD")
    
    if not check_file_exists('best_model.h5'):
        print("❌ Modèle standard non trouvé!")
        print("💡 Entraînez d'abord un modèle")
        return
    
    print("\n🚀 Démarrage de la détection...")
    
    try:
        subprocess.run(['python3', 'webcam_macos_app.py'])
    except KeyboardInterrupt:
        print("\n⚠️  Détection interrompue")
    except Exception as e:
        print(f"\n❌ Erreur: {e}")

def verify_installation():
    """Vérifier l'installation complète"""
    print_header("🔍 VÉRIFICATION COMPLÈTE DE L'INSTALLATION")
    
    all_ok = True
    
    # Dépendances
    if not check_dependencies():
        all_ok = False
    
    # Données
    if not check_data():
        all_ok = False
    
    # Modèles
    available_models = check_models()
    
    # Scripts
    print_header("📜 VÉRIFICATION DES SCRIPTS")
    
    scripts = [
        'train_optimized_cnn.py',
        'webcam_enhanced_detection.py',
        'run_facial_recognition.py',
        'webcam_macos_app.py'
    ]
    
    for script in scripts:
        if check_file_exists(script):
            print(f"✅ {script}")
        else:
            print(f"❌ {script} - MANQUANT")
            all_ok = False
    
    # Résumé
    print_header("📊 RÉSUMÉ")
    
    if all_ok and len(available_models) > 0:
        print("✅ Installation complète et fonctionnelle!")
        print("\n💡 Vous pouvez:")
        print("   - Utiliser la détection en temps réel (option 2)")
        if 'best_model_optimized.h5' not in available_models:
            print("   - Entraîner le modèle optimisé pour de meilleures performances (option 1)")
    elif all_ok:
        print("⚠️  Installation OK mais aucun modèle entraîné")
        print("\n💡 Prochaine étape:")
        print("   - Entraîner un modèle (option 1)")
    else:
        print("❌ Installation incomplète")
        print("\n💡 Vérifiez les erreurs ci-dessus")

def show_guide():
    """Afficher le guide"""
    print_header("📚 GUIDE D'UTILISATION")
    
    guide_file = 'GUIDE_SYSTEME_AMELIORE.md'
    
    if check_file_exists(guide_file):
        print(f"\n📖 Guide disponible: {guide_file}")
        print("\n💡 Pour lire le guide:")
        print(f"   - Ouvrir {guide_file} dans un éditeur de texte")
        print(f"   - Ou: cat {guide_file}")
        
        show = input("\nAfficher un résumé? [o/N]: ").strip().lower()
        
        if show == 'o':
            print("\n" + "=" * 70)
            print("📋 RÉSUMÉ RAPIDE")
            print("=" * 70)
            print("""
🚀 DÉMARRAGE RAPIDE:

1. Entraîner le modèle optimisé:
   python3 train_optimized_cnn.py
   
2. Tester en temps réel:
   python3 webcam_enhanced_detection.py

🎯 AMÉLIORATIONS:
   ✅ Détection multi-méthodes (Haar + validation yeux)
   ✅ Prétraitement avancé (égalisation + débruitage)
   ✅ Architecture CNN optimisée (512 filtres, Dropout, L2)
   ✅ Data augmentation améliorée
   ✅ Gestion déséquilibre de classes

📊 RÉSULTATS ATTENDUS:
   - Accuracy: 75-80% (vs 60-65% ancien)
   - Taux détection: 85-95% (vs 60-70% ancien)
   - Amélioration: +15-20%

📚 DOCUMENTATION COMPLÈTE:
   Voir GUIDE_SYSTEME_AMELIORE.md
            """)
    else:
        print(f"\n❌ Guide non trouvé: {guide_file}")

def main():
    """Fonction principale"""
    try:
        while True:
            choice = main_menu()
            
            if choice == '1':
                train_optimized_model()
            elif choice == '2':
                run_enhanced_detection()
            elif choice == '3':
                run_standard_detection()
            elif choice == '4':
                verify_installation()
            elif choice == '5':
                show_guide()
            elif choice == '6':
                print("\n👋 Au revoir!")
                break
            else:
                print("\n❌ Choix invalide. Veuillez choisir entre 1 et 6.")
            
            input("\n⏸️  Appuyez sur Entrée pour continuer...")
            print("\n" * 2)
    
    except KeyboardInterrupt:
        print("\n\n👋 Au revoir!")
    except Exception as e:
        print(f"\n❌ Erreur inattendue: {e}")

if __name__ == "__main__":
    main()

# Made with Bob
