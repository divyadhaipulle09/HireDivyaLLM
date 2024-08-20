import streamlit as st

# Function to style the app
def apply_custom_styles():
    st.markdown(
        """
        <style>
        .main {
            background-color: #f0f2f6;
            padding: 2rem;
            border-radius: 10px;
        }
        .header {
            text-align: center;
            color: #333;
            padding-bottom: 1rem;
        }
        .card {
            background: white;
            padding: 1rem;
            border-radius: 8px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            margin-bottom: 1rem;
        }
        .card img {
            border-radius: 8px;
        }
        .button {
            display: inline-block;
            padding: 0.5rem 1rem;
            font-size: 16px;
            font-weight: bold;
            color: #fff;
            background-color: #0072B1;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            text-decoration: none;
        }
        .button:hover {
            background-color: #005f8d;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

# Sidebar with professional look
def sidebar():
    LinkedIn_url = 'https://www.linkedin.com/in/divya-dhaipulle/'
    Portfolio_url = 'https://divyadhaipulle09.github.io/divyadhaipullay.github.io/'
    Email = 'ddhaipul@gmail.com'
    Phone = '+1 8125433634'
    
    st.sidebar.markdown(
        "<h1 style='text-align: center; color: #0072B1;'>Divya Dhaipullay</h1>", unsafe_allow_html=True
    )
    st.sidebar.image(r"Me.jpg", use_column_width=True)
    
    st.sidebar.markdown(
        "<p style='text-align: center;'>Machine Learning Engineer | Data Scientist | Data Analyst | AI Engineer</p>",
        unsafe_allow_html=True
    )
    
    st.sidebar.markdown(
        f"<p style='text-align: center;'>Email: <a href='mailto:{Email}'>{Email}</a><br>Phone: {Phone}</p>",
        unsafe_allow_html=True
    )
    
    st.sidebar.markdown(
        "<div style='text-align: center;'>"
        f"<a href='{LinkedIn_url}' target='_blank' class='button'>LinkedIn</a> | "
        f"<a href='{Portfolio_url}' target='_blank' class='button'>Portfolio</a>"
        "</div>",
        unsafe_allow_html=True
    )
    
    st.sidebar.markdown(
        "<p style='text-align: center; font-weight: bold;'>"
        "Actively seeking full-time opportunities.<br>Open to relocation | Can start immediately"
        "</p>",
        unsafe_allow_html=True
    )

# Main app content
def main_content():
    st.markdown(
        "<div class='header'><h1>Welcome to My Professional Portfolio</h1></div>", 
        unsafe_allow_html=True
    )
    
    st.markdown(
        "<div class='card'>"
        "<h2>About Me</h2>"
        "<p>Hello! I'm Divya Dhaipullay, a Machine Learning Engineer with extensive experience in data science and AI. My expertise includes building end-to-end ML systems, working with LLMs, and deploying real-time image recognition systems. I am actively seeking new opportunities where I can contribute my skills and drive success.</p>"
        "</div>",
        unsafe_allow_html=True
    )
    
    st.markdown(
        "<div class='card'>"
        "<h2>My Work</h2>"
        "<p>Check out some of my projects and research work on my <a href='https://divyadhaipulle09.github.io/divyadhaipullay.github.io/' target='_blank' class='button'>Portfolio</a> or view my publications on <a href='https://scholar.google.com/citations?user=bZeCWlwAAAAJ&hl=en&oi=ao' target='_blank' class='button'>Google Scholar</a>.</p>"
        "</div>",
        unsafe_allow_html=True
    )

def main():
    apply_custom_styles()
    sidebar()
    main_content()

if __name__ == '__main__':
    main()
