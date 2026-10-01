from datetime import datetime
from functools import wraps
from pathlib import Path
import os
import uuid

from sqlalchemy import or_

from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    abort,
    send_from_directory,
    flash
)

from flask_sqlalchemy import SQLAlchemy

from flask_login import (
    LoginManager,
    UserMixin,
    login_user,
    logout_user,
    login_required,
    current_user
)

from werkzeug.security import (
    generate_password_hash,
    check_password_hash
)

from werkzeug.utils import secure_filename

from services.ml_service import predict_category


# ==================================================
# APP CONFIGURATION
# ==================================================

BASE_DIR = Path(__file__).resolve().parent

app = Flask(
    __name__,
    template_folder=str(BASE_DIR / "templates"),
    static_folder=str(BASE_DIR / "static")
)

app.config["SECRET_KEY"] = (
    "civicsense-development-secret-key"
)

app.config["SQLALCHEMY_DATABASE_URI"] = (
    "sqlite:///civicsense.db"
)

app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

app.config["MAX_CONTENT_LENGTH"] = (
    5 * 1024 * 1024
)


# ==================================================
# UPLOAD CONFIGURATION
# ==================================================

UPLOAD_FOLDER = (
    BASE_DIR / "static" / "uploads"
)

UPLOAD_FOLDER.mkdir(
    parents=True,
    exist_ok=True
)

app.config["UPLOAD_FOLDER"] = str(
    UPLOAD_FOLDER
)

ALLOWED_IMAGE_EXTENSIONS = {
    "jpg",
    "jpeg",
    "png",
    "webp"
}


# ==================================================
# DATABASE AND LOGIN
# ==================================================

db = SQLAlchemy(app)

login_manager = LoginManager()

login_manager.init_app(app)

login_manager.login_view = "login"


# ==================================================
# HELPER FUNCTIONS
# ==================================================

def allowed_image(filename):
    return (
        "." in filename
        and filename.rsplit(
            ".",
            1
        )[1].lower()
        in ALLOWED_IMAGE_EXTENSIONS
    )


def create_safe_filename(filename):
    clean_name = secure_filename(
        filename
    )

    extension = ""

    if "." in clean_name:
        extension = (
            "."
            + clean_name.rsplit(
                ".",
                1
            )[1].lower()
        )

    return uuid.uuid4().hex + extension


# ==================================================
# DATABASE MODELS
# ==================================================

class User(UserMixin, db.Model):
    id = db.Column(
        db.Integer,
        primary_key=True
    )

    name = db.Column(
        db.String(100),
        nullable=False
    )

    email = db.Column(
        db.String(120),
        unique=True,
        nullable=False
    )

    password_hash = db.Column(
        db.String(255),
        nullable=False
    )

    role = db.Column(
        db.String(30),
        default="citizen",
        nullable=False
    )


class Complaint(db.Model):
    id = db.Column(
        db.Integer,
        primary_key=True
    )

    title = db.Column(
        db.String(150),
        nullable=False
    )

    description = db.Column(
        db.Text,
        nullable=False
    )

    category = db.Column(
        db.String(50),
        default="Other",
        nullable=False
    )

    priority = db.Column(
        db.String(30),
        default="Medium",
        nullable=False
    )

    status = db.Column(
        db.String(30),
        default="Submitted",
        nullable=False
    )

    location = db.Column(
        db.String(200),
        nullable=True
    )

    image_filename = db.Column(
        db.String(255),
        nullable=True
    )

    created_at = db.Column(
        db.DateTime,
        default=datetime.utcnow,
        nullable=False
    )

    updated_at = db.Column(
        db.DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        nullable=False
    )

    user_id = db.Column(
        db.Integer,
        db.ForeignKey("user.id"),
        nullable=False
    )

    predicted_category = db.Column(
        db.String(50),
        nullable=True
    )

    ml_confidence = db.Column(
        db.Float,
        nullable=True
    )


# ==================================================
# LOGIN USER LOADER
# ==================================================

@login_manager.user_loader
def load_user(user_id):
    return db.session.get(
        User,
        int(user_id)
    )


# ==================================================
# ADMIN DECORATOR
# ==================================================

def admin_required(view_function):

    @wraps(view_function)
    @login_required
    def wrapped_view(*args, **kwargs):

        if current_user.role != "admin":
            abort(403)

        return view_function(
            *args,
            **kwargs
        )

    return wrapped_view


# ==================================================
# HOME
# ==================================================

@app.route("/")
def home():
    return render_template(
        "home.html"
    )


# ==================================================
# HEALTH CHECK
# ==================================================

@app.route("/health")
def health():
    return {
        "status": "success",
        "message": "CivicSense backend is running"
    }


# ==================================================
# REGISTER
# ==================================================

@app.route(
    "/register",
    methods=["GET", "POST"]
)
def register():

    if current_user.is_authenticated:
        logout_user()

    if request.method == "POST":

        name = request.form.get(
            "name",
            ""
        ).strip()

        email = request.form.get(
            "email",
            ""
        ).strip().lower()

        password = request.form.get(
            "password",
            ""
        )

        admin_key = request.form.get(
            "admin_key",
            ""
        ).strip()

        if not name:
            flash(
                "Name is required.",
                "error"
            )

            return redirect(
                url_for("register")
            )

        if not email:
            flash(
                "Email is required.",
                "error"
            )

            return redirect(
                url_for("register")
            )

        if not password:
            flash(
                "Password is required.",
                "error"
            )

            return redirect(
                url_for("register")
            )

        if len(password) < 6:
            flash(
                "Password must contain at least 6 characters.",
                "error"
            )

            return redirect(
                url_for("register")
            )

        existing_user = User.query.filter_by(
            email=email
        ).first()

        if existing_user:
            flash(
                "This email is already registered.",
                "error"
            )

            return redirect(
                url_for("register")
            )

        role = "citizen"

        if admin_key == "CIVIC-ADMIN-2026":
            role = "admin"

        user = User(
            name=name,
            email=email,
            password_hash=generate_password_hash(
                password
            ),
            role=role
        )

        db.session.add(user)
        db.session.commit()

        flash(
            "Account created successfully. Please login.",
            "success"
        )

        return redirect(
            url_for("login")
        )

    return render_template(
        "register.html"
    )


# ==================================================
# LOGIN
# ==================================================

@app.route(
    "/login",
    methods=["GET", "POST"]
)
def login():

    if current_user.is_authenticated:

        if current_user.role == "admin":
            return redirect(
                url_for("admin_dashboard")
            )

        return redirect(
            url_for("dashboard")
        )

    if request.method == "POST":

        email = request.form.get(
            "email",
            ""
        ).strip().lower()

        password = request.form.get(
            "password",
            ""
        )

        user = User.query.filter_by(
            email=email
        ).first()

        if user is None:
            flash(
                "No account found with this email.",
                "error"
            )

            return redirect(
                url_for("login")
            )

        password_is_correct = (
            check_password_hash(
                user.password_hash,
                password
            )
        )

        if not password_is_correct:
            flash(
                "Incorrect password.",
                "error"
            )

            return redirect(
                url_for("login")
            )

        login_user(user)

        if user.role == "admin":
            return redirect(
                url_for("admin_dashboard")
            )

        return redirect(
            url_for("dashboard")
        )

    return render_template(
        "login.html"
    )


# ==================================================
# LOGOUT
# ==================================================

@app.route("/logout")
@login_required
def logout():

    logout_user()

    return redirect(
        url_for("login")
    )


# ==================================================
# SERVE COMPLAINT IMAGES
# ==================================================

@app.route(
    "/complaint-image/<path:filename>"
)
@login_required
def complaint_image(filename):

    return send_from_directory(
        app.config["UPLOAD_FOLDER"],
        filename
    )


# ==================================================
# CITIZEN DASHBOARD
# ==================================================

@app.route("/dashboard")
@login_required
def dashboard():

    if current_user.role == "admin":
        return redirect(
            url_for("admin_dashboard")
        )

    complaints = Complaint.query.filter_by(
        user_id=current_user.id
    ).order_by(
        Complaint.created_at.desc()
    ).all()

    return render_template(
        "dashboard.html",
        complaints=complaints
    )


# ==================================================
# CREATE COMPLAINT
# ==================================================

@app.route(
    "/complaints/new",
    methods=["GET", "POST"]
)
@login_required
def new_complaint():

    if current_user.role == "admin":
        return redirect(
            url_for("admin_dashboard")
        )

    if request.method == "POST":

        title = request.form.get(
            "title",
            ""
        ).strip()

        description = request.form.get(
            "description",
            ""
        ).strip()

        category_from_form = request.form.get(
            "category",
            ""
        ).strip()

        priority = request.form.get(
            "priority",
            "Medium"
        ).strip()

        location = request.form.get(
            "location",
            ""
        ).strip()

        image = request.files.get(
            "image"
        )

        if not title:
            return "Issue title is required."

        if len(description) < 20:
            return (
                "Description must contain "
                "at least 20 characters."
            )

        allowed_priorities = [
            "Low",
            "Medium",
            "High",
            "Emergency"
        ]

        if priority not in allowed_priorities:
            priority = "Medium"

        image_filename = None

        if image and image.filename:

            if not allowed_image(
                image.filename
            ):
                return (
                    "Only JPG, JPEG, PNG and WEBP "
                    "images are allowed."
                )

            image_filename = create_safe_filename(
                image.filename
            )

            image_path = os.path.join(
                app.config["UPLOAD_FOLDER"],
                image_filename
            )

            image.save(
                image_path
            )

        predicted_category = "Other"
        confidence = 0.0

        if category_from_form:
            predicted_category = (
                category_from_form
            )

        else:
            try:
                predicted_category, confidence = (
                    predict_category(
                        description
                    )
                )

            except Exception as error:
                print(
                    "ML prediction error:",
                    error
                )

                predicted_category = "Other"
                confidence = 0.0

        complaint = Complaint(
            title=title,
            description=description,
            category=predicted_category,
            priority=priority,
            status="Submitted",
            location=location,
            image_filename=image_filename,
            user_id=current_user.id,
            predicted_category=predicted_category,
            ml_confidence=confidence
        )

        db.session.add(
            complaint
        )

        db.session.commit()

        flash(
            "Complaint submitted successfully.",
            "success"
        )

        return redirect(
            url_for("dashboard")
        )

    return render_template(
        "new_complaint.html"
    )


# ==================================================
# ALL COMPLAINTS
# ==================================================

@app.route("/complaints")
@login_required
def complaint_list():

    search = request.args.get(
        "search",
        ""
    ).strip()

    status_filter = request.args.get(
        "status",
        ""
    ).strip()

    priority_filter = request.args.get(
        "priority",
        ""
    ).strip()

    category_filter = request.args.get(
        "category",
        ""
    ).strip()

    query = Complaint.query

    if search:

        search_pattern = (
            f"%{search}%"
        )

        query = query.filter(
            or_(
                Complaint.title.ilike(
                    search_pattern
                ),
                Complaint.description.ilike(
                    search_pattern
                ),
                Complaint.location.ilike(
                    search_pattern
                ),
                Complaint.category.ilike(
                    search_pattern
                )
            )
        )

    if status_filter:
        query = query.filter_by(
            status=status_filter
        )

    if priority_filter:
        query = query.filter_by(
            priority=priority_filter
        )

    if category_filter:
        query = query.filter_by(
            category=category_filter
        )

    complaints = query.order_by(
        Complaint.created_at.desc()
    ).all()

    all_categories = db.session.query(
        Complaint.category
    ).distinct().order_by(
        Complaint.category.asc()
    ).all()

    categories = [
        category[0]
        for category in all_categories
        if category[0]
    ]

    return render_template(
        "complaints.html",
        complaints=complaints,
        categories=categories,
        search=search,
        status_filter=status_filter,
        priority_filter=priority_filter,
        category_filter=category_filter
    )


# ==================================================
# ADMIN DASHBOARD
# ==================================================

@app.route("/admin/dashboard")
@admin_required
def admin_dashboard():

    complaints = Complaint.query.order_by(
        Complaint.created_at.desc()
    ).all()

    total_complaints = Complaint.query.count()

    submitted_complaints = Complaint.query.filter_by(
        status="Submitted"
    ).count()

    in_progress_complaints = Complaint.query.filter_by(
        status="In Progress"
    ).count()

    resolved_complaints = Complaint.query.filter_by(
        status="Resolved"
    ).count()

    high_priority_complaints = Complaint.query.filter(
        Complaint.priority.in_(
            [
                "High",
                "Emergency"
            ]
        )
    ).count()

    return render_template(
        "admin_dashboard.html",
        complaints=complaints,
        total_complaints=total_complaints,
        submitted_complaints=submitted_complaints,
        in_progress_complaints=in_progress_complaints,
        resolved_complaints=resolved_complaints,
        high_priority_complaints=high_priority_complaints
    )


# ==================================================
# UPDATE COMPLAINT STATUS
# ==================================================

@app.route(
    "/admin/complaints/<int:complaint_id>/status",
    methods=["POST"]
)
@admin_required
def update_complaint_status(
    complaint_id
):

    complaint = db.session.get(
        Complaint,
        complaint_id
    )

    if complaint is None:
        abort(404)

    new_status = request.form.get(
        "status",
        ""
    ).strip()

    allowed_statuses = [
        "Submitted",
        "Assigned",
        "In Progress",
        "Resolved",
        "Closed"
    ]

    if new_status not in allowed_statuses:

        flash(
            "Invalid complaint status.",
            "error"
        )

        return redirect(
            url_for("admin_dashboard")
        )

    complaint.status = new_status
    complaint.updated_at = datetime.utcnow()

    db.session.commit()

    flash(
        "Complaint status updated successfully.",
        "success"
    )

    return redirect(
        url_for("admin_dashboard")
    )


# ==================================================
# ERROR HANDLERS
# ==================================================

@app.errorhandler(403)
def forbidden(error):
    return (
        "<h1>403 - Access Denied</h1>"
        "<p>You are not allowed to view this page.</p>",
        403
    )


@app.errorhandler(413)
def file_too_large(error):
    return (
        "<h1>File Too Large</h1>"
        "<p>Image size must be less than 5 MB.</p>",
        413
    )


# ==================================================
# CREATE TABLES AND RUN APP
# ==================================================

if __name__ == "__main__":

    with app.app_context():
        db.create_all()

    app.run(
        debug=True,
        host="127.0.0.1",
        port=5000
    )