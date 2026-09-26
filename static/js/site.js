// site.js – handles navbar mobile toggle, scroll‑reveal animations, and optional canvas background

document.addEventListener('DOMContentLoaded', () => {
    // Mobile navigation toggle
    const navToggle = document.getElementById('navToggle');
    const navLinks = document.getElementById('navLinks');
    const siteNav = document.getElementById('siteNav');

    if (navToggle && navLinks) {
        navToggle.addEventListener('click', () => {
            navLinks.classList.toggle('open');
        });
    }

    // Add shadow to navbar on scroll
    window.addEventListener('scroll', () => {
        if (window.scrollY > 20) {
            siteNav.classList.add('scrolled');
        } else {
            siteNav.classList.remove('scrolled');
        }
    });

    // Simple scroll‑reveal implementation
    const revealElements = document.querySelectorAll('.reveal');
    const revealOnScroll = () => {
        const viewportHeight = window.innerHeight;
        revealElements.forEach(el => {
            const rect = el.getBoundingClientRect();
            if (rect.top < viewportHeight - 60) {
                el.classList.add('visible');
            }
        });
    };
    revealOnScroll(); // initial check
    window.addEventListener('scroll', revealOnScroll);

    // Optional animated canvas background – lightweight particles
    const canvas = document.getElementById('bgCanvas');
    if (canvas && canvas.getContext) {
        const ctx = canvas.getContext('2d');
        let width, height;
        const particles = [];
        const particleCount = 80;
        const maxSpeed = 0.6;
        const connectDist = 120;

        const resize = () => {
            width = canvas.width = window.innerWidth;
            height = canvas.height = window.innerHeight;
        };
        resize();
        window.addEventListener('resize', resize);

        class Particle {
            constructor() {
                this.x = Math.random() * width;
                this.y = Math.random() * height;
                this.vx = (Math.random() - 0.5) * maxSpeed;
                this.vy = (Math.random() - 0.5) * maxSpeed;
                this.size = 2 + Math.random() * 2;
            }
            update() {
                this.x += this.vx;
                this.y += this.vy;
                if (this.x < 0 || this.x > width) this.vx *= -1;
                if (this.y < 0 || this.y > height) this.vy *= -1;
            }
            draw() {
                ctx.fillStyle = 'rgba(45,212,191,0.6)';
                ctx.beginPath();
                ctx.arc(this.x, this.y, this.size, 0, Math.PI * 2);
                ctx.fill();
            }
        }

        const initParticles = () => {
            particles.length = 0;
            for (let i = 0; i < particleCount; i++) {
                particles.push(new Particle());
            }
        };
        initParticles();

        const connect = () => {
            for (let i = 0; i < particles.length; i++) {
                for (let j = i + 1; j < particles.length; j++) {
                    const dx = particles[i].x - particles[j].x;
                    const dy = particles[i].y - particles[j].y;
                    const dist = Math.sqrt(dx * dx + dy * dy);
                    if (dist < connectDist) {
                        const opacity = 1 - dist / connectDist;
                        ctx.strokeStyle = `rgba(45,212,191,${opacity * 0.2})`;
                        ctx.lineWidth = 1;
                        ctx.beginPath();
                        ctx.moveTo(particles[i].x, particles[i].y);
                        ctx.lineTo(particles[j].x, particles[j].y);
                        ctx.stroke();
                    }
                }
            }
        };

        const animate = () => {
            ctx.clearRect(0, 0, width, height);
            particles.forEach(p => {
                p.update();
                p.draw();
            });
            connect();
            requestAnimationFrame(animate);
        };
        animate();
    }
});
