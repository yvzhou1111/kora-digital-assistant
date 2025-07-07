import { User } from './user';
import { DigitalTwin } from './twin';

export interface Expert {
  id: string;
  userId: string;
  bio: string;
  title: string;
  specializations: string[];
  expertise: ExpertiseArea[];
  education: Education[];
  experience: Experience[];
  certifications: Certification[];
  publications: Publication[];
  consultationFee: {
    amount: number;
    currency: string;
    unit: 'hour' | 'session' | 'project';
  };
  availability: Availability;
  rating: number;
  reviewCount: number;
  verified: boolean;
  featuredSkills: string[];
  socialProof: SocialProof;
  twinId?: string;
}

export interface ExpertiseArea {
  id: string;
  name: string;
  level: number; // 1-5
  yearsOfExperience: number;
  description?: string;
}

export interface Education {
  id: string;
  institution: string;
  degree: string;
  field: string;
  startDate: Date;
  endDate?: Date;
  isOngoing: boolean;
  description?: string;
}

export interface Experience {
  id: string;
  company: string;
  title: string;
  location?: string;
  startDate: Date;
  endDate?: Date;
  isOngoing: boolean;
  description: string;
  achievements?: string[];
}

export interface Certification {
  id: string;
  name: string;
  issuingOrganization: string;
  issueDate: Date;
  expirationDate?: Date;
  credentialId?: string;
  credentialUrl?: string;
}

export interface Publication {
  id: string;
  title: string;
  publisher: string;
  publishDate: Date;
  url?: string;
  description?: string;
  coAuthors?: string[];
}

export interface Availability {
  schedule: {
    [day: string]: TimeSlot[]; // day: 'monday', 'tuesday', etc.
  };
  timezone: string;
  exceptions: {
    date: Date;
    available: boolean;
    slots?: TimeSlot[];
  }[];
}

export interface TimeSlot {
  start: string; // HH:MM format
  end: string; // HH:MM format
}

export interface SocialProof {
  followers: number;
  linkedInConnections?: number;
  githubStars?: number;
  websiteVisits?: number;
  featuredClients?: string[];
}

export interface ConsultationService {
  id: string;
  expertId: string;
  title: string;
  description: string;
  category: ServiceCategory;
  format: ServiceFormat;
  duration: number; // In minutes
  price: {
    amount: number;
    currency: string;
  };
  isActive: boolean;
  deliverables?: string[];
  prerequisites?: string[];
  faqs?: {
    question: string;
    answer: string;
  }[];
}

export enum ServiceCategory {
  CAREER_ADVICE = 'career_advice',
  TECHNICAL_CONSULTATION = 'technical_consultation',
  PORTFOLIO_REVIEW = 'portfolio_review',
  MOCK_INTERVIEW = 'mock_interview',
  MENTORSHIP = 'mentorship',
  TRAINING = 'training',
  PROJECT_REVIEW = 'project_review',
  OTHER = 'other',
}

export enum ServiceFormat {
  ONE_ON_ONE = 'one_on_one',
  GROUP = 'group',
  WORKSHOP = 'workshop',
  ASYNC = 'async',
  TEXT_BASED = 'text_based',
}

export interface ConsultationSession {
  id: string;
  serviceId: string;
  expertId: string;
  clientId: string;
  status: SessionStatus;
  scheduledAt: Date;
  completedAt?: Date;
  duration: number; // In minutes
  notes?: {
    client?: string;
    expert?: string;
  };
  feedback?: {
    rating: number; // 1-5
    comment?: string;
  };
  payment: {
    amount: number;
    currency: string;
    status: 'pending' | 'paid' | 'refunded';
  };
}

export enum SessionStatus {
  SCHEDULED = 'scheduled',
  IN_PROGRESS = 'in_progress',
  COMPLETED = 'completed',
  CANCELLED = 'cancelled',
  NO_SHOW = 'no_show',
}

export interface KnowledgeBaseItem {
  id: string;
  expertId: string;
  title: string;
  content: string;
  category: string;
  tags: string[];
  visibility: 'public' | 'private' | 'twin_only';
  createdAt: Date;
  updatedAt: Date;
  views: number;
  helpfulRating: number;
}

export interface ExpertState {
  experts: Expert[];
  selectedExpert: Expert | null;
  consultationServices: ConsultationService[];
  sessions: ConsultationSession[];
  knowledgeBase: KnowledgeBaseItem[];
  isLoading: boolean;
  error: string | null;
} 