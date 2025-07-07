import { User, UserRole } from './user';
import { DigitalTwin } from './twin';

export interface Job {
  id: string;
  employerId: string;
  title: string;
  company: string;
  location: string;
  description: string;
  requirements: string[];
  responsibilities: string[];
  employmentType: EmploymentType;
  experienceLevel: ExperienceLevel;
  salaryRange?: {
    min: number;
    max: number;
    currency: string;
  };
  skills: string[];
  benefits?: string[];
  status: JobStatus;
  createdAt: Date;
  updatedAt: Date;
  expiresAt?: Date;
  views: number;
  applications: number;
}

export enum EmploymentType {
  FULL_TIME = 'full_time',
  PART_TIME = 'part_time',
  CONTRACT = 'contract',
  TEMPORARY = 'temporary',
  INTERNSHIP = 'internship',
  FREELANCE = 'freelance',
}

export enum ExperienceLevel {
  ENTRY = 'entry',
  MID = 'mid',
  SENIOR = 'senior',
  LEAD = 'lead',
  EXECUTIVE = 'executive',
}

export enum JobStatus {
  DRAFT = 'draft',
  ACTIVE = 'active',
  PAUSED = 'paused',
  CLOSED = 'closed',
  EXPIRED = 'expired',
}

export interface JobApplication {
  id: string;
  jobId: string;
  userId: string;
  twinId?: string;
  status: ApplicationStatus;
  coverLetter?: string;
  resume?: string;
  answers?: { [key: string]: string }; // For screening questions
  createdAt: Date;
  updatedAt: Date;
  notes?: string; // Employer's notes
  feedback?: string; // Feedback to candidate
  interviews?: Interview[];
}

export enum ApplicationStatus {
  SUBMITTED = 'submitted',
  UNDER_REVIEW = 'under_review',
  TWIN_SCREENING = 'twin_screening',
  SHORTLISTED = 'shortlisted',
  INTERVIEW_SCHEDULED = 'interview_scheduled',
  INTERVIEW_COMPLETED = 'interview_completed',
  OFFER_EXTENDED = 'offer_extended',
  HIRED = 'hired',
  REJECTED = 'rejected',
  WITHDRAWN = 'withdrawn',
}

export interface Interview {
  id: string;
  applicationId: string;
  jobId: string;
  candidateId: string;
  employerId: string;
  scheduledAt: Date;
  duration: number; // In minutes
  type: InterviewType;
  location?: string; // Physical location or URL
  status: InterviewStatus;
  notes?: string;
  feedback?: {
    technical: number; // 1-5
    communication: number; // 1-5
    cultural: number; // 1-5
    overall: number; // 1-5
    comments: string;
  };
}

export enum InterviewType {
  PHONE = 'phone',
  VIDEO = 'video',
  IN_PERSON = 'in_person',
  TECHNICAL = 'technical',
  ASSESSMENT = 'assessment',
}

export enum InterviewStatus {
  SCHEDULED = 'scheduled',
  COMPLETED = 'completed',
  CANCELLED = 'cancelled',
  NO_SHOW = 'no_show',
  RESCHEDULED = 'rescheduled',
}

export interface TwinJobInteraction {
  id: string;
  jobId: string;
  twinId: string;
  type: InteractionType;
  questions: string[];
  answers: string[];
  feedback?: string;
  score?: number; // 0-100
  createdAt: Date;
  duration: number; // In seconds
}

export enum InteractionType {
  SCREENING = 'screening',
  MOCK_INTERVIEW = 'mock_interview',
  JOB_CHAT = 'job_chat',
}

export interface JobState {
  jobs: Job[];
  selectedJob: Job | null;
  applications: JobApplication[];
  isLoading: boolean;
  error: string | null;
  filters: JobFilters;
  pagination: Pagination;
}

export interface JobFilters {
  title?: string;
  location?: string;
  employmentType?: EmploymentType[];
  experienceLevel?: ExperienceLevel[];
  skills?: string[];
  salaryMin?: number;
  salaryMax?: number;
}

export interface Pagination {
  page: number;
  limit: number;
  total: number;
} 