// get the ninja-keys element
const ninja = document.querySelector('ninja-keys');

// add the home and posts menu items
ninja.data = [{
    id: "nav-about",
    title: "about",
    section: "Navigation",
    handler: () => {
      window.location.href = "/";
    },
  },{id: "nav-blog",
          title: "blog",
          description: "",
          section: "Navigation",
          handler: () => {
            window.location.href = "/blog/";
          },
        },{id: "nav-publications",
          title: "publications",
          description: "kept in sync with my Google Scholar profile, in the same order.",
          section: "Navigation",
          handler: () => {
            window.location.href = "/publications/";
          },
        },{id: "nav-cv",
          title: "cv",
          description: "Scroll for a long version of my cv, or download the pdf for a shorter one.",
          section: "Navigation",
          handler: () => {
            window.location.href = "/cv/";
          },
        },{id: "post-notes-on-world-models",
        
          title: "Notes on World Models",
        
        description: "Some notes I took at EPFL on world models from C. Bunne’s course. It is still WIP, some references (e.g., for images) are missing.",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2026/notes-on-world-models/";
          
        },
      },{id: "post-notes-on-transformers",
        
          title: "Notes on Transformers",
        
        description: "Some notes I took at EPFL on transformers from C. Bunne’s course. It is still WIP, some references (e.g., for images) are missing.",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2026/notes-on-transformers/";
          
        },
      },{id: "post-notes-on-diffusion-models",
        
          title: "Notes on Diffusion Models",
        
        description: "Some notes I took at EPFL on diffusion models from C. Bunne’s course. It is still WIP, some references (e.g., for images) are missing.",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2026/notes-on-diffusion-models/";
          
        },
      },{id: "books-the-godfather",
          title: 'The Godfather',
          description: "",
          section: "Books",handler: () => {
              window.location.href = "/books/the_godfather/";
            },},{id: "news-our-agent-took-1st-place-out-of-270-students-in-the-intelligent-systems-competition-we-built-a-schnapsen-playing-ai-that-dominated-the-leaderboard-super-fun-project",
          title: 'Our agent took 1st place out of 270 students in the Intelligent Systems...',
          description: "",
          section: "News",},{id: "news-honored-to-receive-the-khmw-young-talent-incentive-award-given-annually-to-the-top-performing-student-in-each-dutch-research-university-grateful-for-the-recognition",
          title: 'Honored to receive the KHMW Young Talent Incentive Award🏅. Given annually to the...',
          description: "",
          section: "News",},{id: "news-started-working-as-a-technology-assistant-at-the-network-institute-supporting-projects-in-xr-motion-capture-and-unity-development-loved-presenting-at-surf-xr-on-tour",
          title: 'Started working as a Technology Assistant at the Network Institute Supporting projects in...',
          description: "",
          section: "News",},{id: "news-our-work-cicero-a-gpt-2-based-writing-assistant-for-e-justice-was-accepted-at-caise-2023-great-collaboration-with-sapienza-s-legal-tech-group-️",
          title: 'Our work CICERO, a GPT-2-based writing assistant for e-justice, was accepted at CAiSE...',
          description: "",
          section: "News",},{id: "news-started-my-summer-research-internship-at-eth-zürich-lre-lab-applying-causal-inference-to-uncover-reasoning-shortcuts-in-llms",
          title: 'Started my summer research internship at ETH Zürich (LRE Lab) 🇨🇭 Applying causal...',
          description: "",
          section: "News",},{id: "news-started-my-research-position-at-gladia-working-on-model-merging-interpretability-and-ability-estimation-for-llms-grateful-for-the-opportunity-and-the-team",
          title: 'Started my research position at GLADIA! Working on model merging, interpretability, and ability...',
          description: "",
          section: "News",},{id: "news-our-paper-do-sparse-autoencoders-transfer-across-base-and-finetuned-llms-was-presented-at-the-neurips-2024-unireps-workshop-fun-results-on-representation-drift-and-transferability",
          title: 'Our paper Do Sparse Autoencoders Transfer Across Base and Finetuned LLMs?was presented at...',
          description: "",
          section: "News",},{id: "news-our-work-on-activation-patching-for-interpretable-steering-in-music-generation-is-out-check-how-steering-tokens-shape-musical-structure",
          title: 'Our work on Activation Patching for Interpretable Steering in Music Generation is out....',
          description: "",
          section: "News",},{id: "news-merge-was-accepted-at-icml-2025-excited-to-share-our-work-on-fast-evolutionary-merging-on-consumer-gpus",
          title: 'MERGE³ was accepted at ICML 2025! Excited to share our work on fast...',
          description: "",
          section: "News",},{id: "news-our-paper-mergenetic-a-simple-evolutionary-model-merging-library-was-accepted-at-acl-2025-system-demonstrations",
          title: 'Our paper Mergenetic: a Simple Evolutionary Model Merging Library was accepted at ACL...',
          description: "",
          section: "News",},{id: "news-excited-to-be-joining-ista-this-summer-as-a-research-intern-in-the-locatello-group-working-on-multimodal-foundation-models-and-causal-learning",
          title: 'Excited to be joining ISTA this summer as a Research Intern in the...',
          description: "",
          section: "News",},{id: "news-new-preprint-out-language-models-are-injective-and-hence-invertible-we-prove-that-llm-representations-are-injective-and-present-the-first-exact-inversion-algorithm-the-announcement-blew-up-with-5m-views-on-twitter-check-it-out",
          title: 'New preprint out! Language Models Are Injective and Hence Invertible We prove that...',
          description: "",
          section: "News",},{id: "news-our-paper-language-models-are-injective-and-hence-invertible-has-been-accepted-at-iclr-2026",
          title: 'Our paper Language Models Are Injective and Hence Invertible has been accepted at...',
          description: "",
          section: "News",},{id: "news-our-paper-exploratory-causal-inference-in-saence-received-an-oral-presentation-at-iclr-2026",
          title: 'Our paper Exploratory Causal Inference in SAEnce received an oral presentation 🏆 at...',
          description: "",
          section: "News",},{id: "news-proposer-of-a-research-team-at-rome-ai-safety-rais-funded-by-coefficient-giving-with-80k-to-develop-disentangled-composable-steering-vectors-for-fine-grained-behavioral-control-of-llms",
          title: 'Proposer of a research team at Rome AI Safety (RAIS), funded by Coefficient...',
          description: "",
          section: "News",},{id: "news-our-paper-multi-objective-evolutionary-merging-enables-efficient-reasoning-models-has-been-accepted-at-colm-2026",
          title: 'Our paper Multi-objective Evolutionary Merging Enables Efficient Reasoning Models has been accepted at...',
          description: "",
          section: "News",},{id: "projects-project-1",
          title: 'project 1',
          description: "with background image",
          section: "Projects",handler: () => {
              window.location.href = "/projects/1_project/";
            },},{id: "projects-project-2",
          title: 'project 2',
          description: "a project with a background image and giscus comments",
          section: "Projects",handler: () => {
              window.location.href = "/projects/2_project/";
            },},{id: "projects-project-3-with-very-long-name",
          title: 'project 3 with very long name',
          description: "a project that redirects to another website",
          section: "Projects",handler: () => {
              window.location.href = "/projects/3_project/";
            },},{id: "projects-project-4",
          title: 'project 4',
          description: "another without an image",
          section: "Projects",handler: () => {
              window.location.href = "/projects/4_project/";
            },},{id: "projects-project-5",
          title: 'project 5',
          description: "a project with a background image",
          section: "Projects",handler: () => {
              window.location.href = "/projects/5_project/";
            },},{id: "projects-project-6",
          title: 'project 6',
          description: "a project with no image",
          section: "Projects",handler: () => {
              window.location.href = "/projects/6_project/";
            },},{id: "projects-project-7",
          title: 'project 7',
          description: "with background image",
          section: "Projects",handler: () => {
              window.location.href = "/projects/7_project/";
            },},{id: "projects-project-8",
          title: 'project 8',
          description: "an other project with a background image and giscus comments",
          section: "Projects",handler: () => {
              window.location.href = "/projects/8_project/";
            },},{id: "projects-project-9",
          title: 'project 9',
          description: "another project with an image 🎉",
          section: "Projects",handler: () => {
              window.location.href = "/projects/9_project/";
            },},{
        id: 'social-email',
        title: 'email',
        section: 'Socials',
        handler: () => {
          window.open("mailto:%74%6F%6D%6D%61%73%6F.%6D%65%6E%63%61%74%74%69%6E%69@%65%70%66%6C.%63%68", "_blank");
        },
      },{
        id: 'social-github',
        title: 'GitHub',
        section: 'Socials',
        handler: () => {
          window.open("https://github.com/tommasomncttn", "_blank");
        },
      },{
        id: 'social-linkedin',
        title: 'LinkedIn',
        section: 'Socials',
        handler: () => {
          window.open("https://www.linkedin.com/in/tommasomencattini", "_blank");
        },
      },{
        id: 'social-scholar',
        title: 'Google Scholar',
        section: 'Socials',
        handler: () => {
          window.open("https://scholar.google.com/citations?user=nJuzaPsAAAAJ", "_blank");
        },
      },{
        id: 'social-x',
        title: 'X',
        section: 'Socials',
        handler: () => {
          window.open("https://twitter.com/tommaso_mncttn", "_blank");
        },
      },];
