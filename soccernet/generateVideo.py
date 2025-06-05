import cv2
import os


def generate_video_from_frames(folder_path, output_filename, frame_rate=30):
    # Vérifier si le dossier existe
    if not os.path.exists(folder_path):
        print(f"❌ Le dossier '{folder_path}' n'existe pas.")
        return

    # Lister les fichiers dans le dossier
    frame_files = sorted([f for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))])

    # Vérifier qu'il y a bien 750 frames
    if len(frame_files) != 750:
        print(f"❌ Nombre de frames incorrect : {len(frame_files)} au lieu de 750.")
        return

    # Lire la première image pour obtenir les dimensions
    first_frame_path = os.path.join(folder_path, frame_files[0])
    first_frame = cv2.imread(first_frame_path)
    if first_frame is None:
        print(f"❌ Impossible de lire la première image : {first_frame_path}")
        return

    height, width, channels = first_frame.shape

    # Initialiser l'objet vidéo
    output_path = f"{output_filename}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec vidéo
    video_writer = cv2.VideoWriter(output_path, fourcc, frame_rate, (width, height))

    print(f"✅ Génération de la vidéo : {output_path}")

    # Ajouter chaque frame à la vidéo
    for frame_file in frame_files:
        frame_path = os.path.join(folder_path, frame_file)
        frame = cv2.imread(frame_path)

        if frame is None:
            print(f"⚠️ Image manquante ou corrompue : {frame_path}")
            continue

        video_writer.write(frame)

    # Libérer les ressources
    video_writer.release()
    print(f"🎬 Vidéo générée avec succès : {output_path}")


# Utilisation
data_path = "data/SoccerNetGS/train/SNGS-062/img1"
generate_video_from_frames(data_path, "output_video_for_spotting")
