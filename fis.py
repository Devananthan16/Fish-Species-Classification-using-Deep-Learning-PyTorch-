import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F
from IPython.display import display
from tqdm import tqdm
from streamlit_option_menu import option_menu
from torchvision import models

st.set_page_config(page_title="Fish Species Classifier", page_icon=":fish:", layout="wide")

print("Done")

option=option_menu("Main menu", ["Cnn", "Vgg16","Resnet50","Mobilenet_v2","densenet"], 
    icons=['house', 'book'], menu_icon="cast", default_index=0, orientation="horizontal")

if option== "Cnn":
    st.title("Fish Species Classifier Using CNN")

    class cnn(nn.Module):
        def __init__(self):
            super().__init__()
            #convolutional layer
            self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
            self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
            
            #fully connected layer
            self.fc1 =nn.Linear(in_features = 64*56*56, out_features=128)
            self.fc2 = nn.Linear(in_features=128, out_features=11)

        def forward(self, x):
            cl1 = self.pool(torch.relu(self.conv1(x)))
            cl2 = self.pool(torch.relu(self.conv2(cl1)))
            flatten_layer = cl2.view(cl2.size(0), -1)
            fully_connected1 = torch.relu(self.fc1(flatten_layer))
            fully_connected2 = self.fc2(fully_connected1)
            return fully_connected2
        
    class_names = [
        'animal fish',
        'animal fish bass',
        'fish sea_food black_sea_sprat',
        'fish sea_food gilt_head_bream',
        'fish sea_food hourse_mackerel',
        'fish sea_food red_mullet',
        'fish sea_food red_sea_bream',
        'fish sea_food sea_bass',
        'fish sea_food shrimp',
        'fish sea_food striped_red_mullet',
        'fish sea_food trout'
    ]
    model = cnn()

    model.load_state_dict(torch.load('cnn_fish_model.pth',map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu')))
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),  
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


    col1, col2 = st.columns(2)
    with col1:
        st.title("Fish Species Classifier")
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        
    
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, caption='Uploaded Image.', use_container_width=True)
            st.write("")
            st.write("Classifying...")
            input_tensor = transform(image).unsqueeze(0).to(device)
            with torch.no_grad():
                output = model(input_tensor)
                _, predicted = torch.max(output, 1)
                predicted_class = class_names[predicted.item()]
        
    with col2:

        st.title("Cnn Model")
        if uploaded_file is not None:
            st.success(f"Predicted Class: {predicted_class}") 
            probab_scores = F.softmax(output, dim=1)
            top5_prob, top5_catid = torch.topk(probab_scores, 5)
            st.subheader("Top 5 Predictions:")
            for i in range(top5_prob.size(1)):
                st.info(f"{class_names[top5_catid[0][i]]}: {top5_prob[0][i].item()*100:.2f}%")

elif option=="Vgg16":
    st.title("Fish Species Classifier Using VGG16")
    
    vgg16 = models.vgg16(pretrained=False)
    vgg16.classifier[6] = nn.Linear(in_features=4096, out_features=11)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    class_names = [
    'animal fish',
    'animal fish bass',
    'fish sea_food black_sea_sprat',
    'fish sea_food gilt_head_bream',
    'fish sea_food hourse_mackerel',
    'fish sea_food red_mullet',
    'fish sea_food red_sea_bream',
    'fish sea_food sea_bass',
    'fish sea_food shrimp',
    'fish sea_food striped_red_mullet',
    'fish sea_food trout'
]

    vgg16.load_state_dict(torch.load('vgg16_fish_model.pth',map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu')))
    vgg16.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vgg16.to(device)
    def predict_image_vgg16(img_path):
        image = Image.open(img_path).convert('RGB')
        display(image)  # Show the image

        input_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = vgg16(input_tensor)
            _, predicted_idx = torch.max(output, 1)

        predicted_class = class_names[predicted_idx.item()]
        print(f"\nPredicted Class: {predicted_class}")
        return predicted_class
    col1, col2 = st.columns(2)
    with col1:   
        st.title("Fish Species Classifier")
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, caption='Uploaded Image.', use_container_width=True)
            st.write("")
            st.write("Classifying...")
            input_tensor = transform(image).unsqueeze(0).to(device)
            with torch.no_grad():
                output = vgg16(input_tensor)
                _, predicted = torch.max(output, 1)
                predicted_class = class_names[predicted.item()]
    with col2:
        st.title("Vgg16 Model")
        if uploaded_file is not None:
            st.success(f"Predicted Class: {predicted_class}")
            probab_scores = F.softmax(output, dim=1)
            top5_prob, top5_catid = torch.topk(probab_scores, 5)
            st.subheader("Top 5 Predictions:")
            for i in range(top5_prob.size(1)):
                st.info(f"{class_names[top5_catid[0][i]]}: {top5_prob[0][i].item()*100:.2f}%")

elif option=="Resnet50":
    st.title("Fish Species Classifier Using Resnet50")
    
    resnet50=models.resnet50(pretrained=False)
    for param in resnet50.parameters():
        param.requires_grad=False
    num_features=resnet50.fc.in_features
    resnet50.fc=nn.Linear(num_features,11)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),  
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    class_names = [
        'animal fish',
        'animal fish bass',
        'fish sea_food black_sea_sprat',
        'fish sea_food gilt_head_bream',
        'fish sea_food hourse_mackerel',
        'fish sea_food red_mullet',
        'fish sea_food red_sea_bream',
        'fish sea_food sea_bass',
        'fish sea_food shrimp',
        'fish sea_food striped_red_mullet',
        'fish sea_food trout'
    ]


    resnet50.load_state_dict(torch.load('resnet50_fish_model1.pth',map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu')))
    resnet50.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    resnet50.to(device)

    def predict_image_resnet50(img_path):
        image = Image.open(img_path).convert('RGB')
        display(image)  # Show the image

        input_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = resnet50(input_tensor)
            _, predicted_idx = torch.max(output, 1)

        predicted_class = class_names[predicted_idx.item()]
        print(f"\nPredicted Class: {predicted_class}")
        return predicted_class

    col1, col2 = st.columns(2)
    with col1:        
        st.title("Fish Species Classifier")
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, caption='Uploaded Image.', use_container_width=True)
            st.write("")
            st.write("Classifying...")
            input_tensor = transform(image).unsqueeze(0).to(device)
            with torch.no_grad():
                output = resnet50(input_tensor)
                _, predicted = torch.max(output, 1)
                predicted_class = class_names[predicted.item()]
    with col2:
        
        st.title("Resnet50 Model")
        if uploaded_file is not None:
            st.success(f"Predicted Class: {predicted_class}")
            probab_scores = F.softmax(output, dim=1)
            top5_prob, top5_catid = torch.topk(probab_scores, 5)
            st.subheader("Top 5 Predictions:")
            for i in range(top5_prob.size(1)):
                st.info(f"{class_names[top5_catid[0][i]]}: {top5_prob[0][i].item()*100:.2f}%")
            
elif option=="Mobilenet_v2":
    st.title("Fish Species Classifier Using Mobilenet_v2")
    
    mobilenet_v2=models.mobilenet_v2(pretrained=False)
    for param in mobilenet_v2.parameters():
        param.requires_grad=False
    num_features=mobilenet_v2.classifier[1].in_features
    mobilenet_v2.classifier[1]=nn.Linear(num_features,11)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),  
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    class_names = [
        'animal fish',
        'animal fish bass',
        'fish sea_food black_sea_sprat',
        'fish sea_food gilt_head_bream',
        'fish sea_food hourse_mackerel',
        'fish sea_food red_mullet',
        'fish sea_food red_sea_bream',
        'fish sea_food sea_bass',
        'fish sea_food shrimp',
        'fish sea_food striped_red_mullet', 
        'fish sea_food trout'
    ]
    mobilenet_v2.load_state_dict(torch.load('mobile_net1.pth',map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu')))
    mobilenet_v2.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mobilenet_v2.to(device) 

    def predict_image_mobilenet_v2(img_path):
        image = Image.open(img_path).convert('RGB')
        display(image)  # Show the image

        input_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = mobilenet_v2(input_tensor)
            _, predicted_idx = torch.max(output, 1)

        predicted_class = class_names[predicted_idx.item()]
        print(f"\nPredicted Class: {predicted_class}")
        return predicted_class
    
    col1, col2 = st.columns(2)
    with col1:
        st.title("Fish Species Classifier")
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, caption='Uploaded Image.', use_container_width=True)
            st.write("")
            st.write("Classifying...")
            input_tensor = transform(image).unsqueeze(0).to(device)
            with torch.no_grad():
                output = mobilenet_v2(input_tensor)
                _, predicted = torch.max(output, 1)
                predicted_class = class_names[predicted.item()]
    with col2:
        st.title("Mobilenet_v2 Model")
        if uploaded_file is not None:
            st.success(f"Predicted Class: {predicted_class}")
            probab_scores = F.softmax(output, dim=1)
            top5_prob, top5_catid = torch.topk(probab_scores, 5)
            st.subheader("Top 5 Predictions:")
            for i in range(top5_prob.size(1)):
                st.info(f"{class_names[top5_catid[0][i]]}: {top5_prob[0][i].item()*100:.2f}%")




elif option=="densenet":
    st.title("Fish Species Classifier Using densenet")
    
    densenet=models.densenet121(pretrained=False)
    for param in densenet.parameters():
        param.requires_grad=False
    num_features=densenet.classifier.in_features
    densenet.classifier=nn.Linear(num_features,11)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),  
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    class_names = [
        'animal fish',
        'animal fish bass',
        'fish sea_food black_sea_sprat',
        'fish sea_food gilt_head_bream',
        'fish sea_food hourse_mackerel',
        'fish sea_food red_mullet',
        'fish sea_food red_sea_bream',
        'fish sea_food sea_bass',
        'fish sea_food shrimp',
        'fish sea_food striped_red_mullet', 
        'fish sea_food trout'
    ]
    state_dict = torch.load('densenet1.pth', map_location='cpu')
    densenet.load_state_dict(state_dict, strict=False)
    densenet.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    densenet.to(device) 

    def predict_image_densenet(img_path):
        image = Image.open(img_path).convert('RGB')
        display(image)  # Show the image

        input_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = densenet(input_tensor)
            _, predicted_idx = torch.max(output, 1)

        predicted_class = class_names[predicted_idx.item()]
        print(f"\nPredicted Class: {predicted_class}")
        return predicted_class
    
    col1, col2 = st.columns(2)
    with col1:
        st.title("Fish Species Classifier")
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, caption='Uploaded Image.', use_container_width=True)
            st.write("")
            st.write("Classifying...")
            input_tensor = transform(image).unsqueeze(0).to(device)
            with torch.no_grad():
                output = densenet(input_tensor)
                _, predicted = torch.max(output, 1)
                predicted_class = class_names[predicted.item()]
    with col2:
        st.title("Densenet Model")
        if uploaded_file is not None:
            st.success(f"Predicted Class: {predicted_class}")
            probab_scores = F.softmax(output, dim=1)
            top5_prob, top5_catid = torch.topk(probab_scores, 5)
            st.subheader("Top 5 Predictions:")
            for i in range(top5_prob.size(1)):
                st.info(f"{class_names[top5_catid[0][i]]}: {top5_prob[0][i].item()*100:.2f}%")


        
        