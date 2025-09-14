import cv2

# Carrega o classificador Haar Cascade pré-treinado para detecção de rostos
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Checa se o arquivo XML foi carregado corretamente
if face_cascade.empty():
    print("Erro: O arquivo haarcascade_frontalface_default.xml não foi carregado. Verifique o caminho.")
    exit()

# Inicia a captura de vídeo da webcam (geralmente 0 para a câmera padrão)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Erro: Não foi possível acessar a webcam.")
    exit()

cv2.waitKey(1)

while True:
    # Lê um frame da webcam
    ret, frame = cap.read()
    if not ret:
        print("Erro: Não foi possível ler o frame da webcam.")
        break

    # Converte o frame para escala de cinza
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detecta rostos na imagem em escala de cinza
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    # Para cada rosto detectado
    for (x, y, w, h) in faces:
        # Desenha um retângulo vermelho (BGR: 0,0,255)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
        # Adiciona o texto "Rosto Detectado" acima do retângulo
        font = cv2.FONT_HERSHEY_SIMPLEX
        texto = "Rosto Detectado"
        tamanho_texto, _ = cv2.getTextSize(texto, font, 0.8, 2)
        text_x = x + (w - tamanho_texto[0]) // 2
        text_y = y - 10 if y - 10 > 20 else y + h + 30
        cv2.putText(frame, texto, (text_x, text_y), font, 0.8, (0, 0, 255), 2, cv2.LINE_AA)

    # Exibe o frame com os retângulos e texto
    cv2.imshow('Detector de Rosto com Haar Cascade', frame)

    # Sai do loop ao pressionar 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libera a webcam e destrói todas as janelas
cap.release()
cv2.destroyAllWindows()